import numpy as np
from collections import namedtuple
from itertools import starmap, repeat
from molecules.ml.unsupervised.cluster import optics_clustering

# TODO: use h5py
#       https://stackoverflow.com/questions/25655588/incremental-writes-to-hdf5-with-h5py?rq=1
#       https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py#47074545
class State:
    def __init__(self):
        #State:    set of all discovered points (latent embeddings)
        #          on the landscape
        self.data = np.array([])

    def save(self, fname):
        np.save(self.data, fname)

    def load(self, fname):
        self.data = np.load(fname)

    def append(self, structures):
        """
        Effects
        -------
        Appends structures to data

        """
        self.data = np.concatenate((self.data, structures))






class REAP:
    """
    Environement: landscape that is to be explored
    State:        set of all discovered points (latent embeddings)
                  on the landscape
    Action:       choosing protein structures to run more simulations on
    """
    def __init__(self, num_order_params, least_samples=5, prior_weights=None):

        # CVAE latent dimension
        self.num_order_params = num_order_params

        # Number of clusters to examine when computing rewards
        self.least_samples = least_samples

        # Same size as the number of order parameters. The ith weight
        # represents the importance of the ith order parameter.
        self.weights = self._init_weights(prior_weights)

        self.state = State()

        Stats = namedtuple('Stats', ['mean', 'stdev'])
        self.stats = dict((op, Stats(0., 1.)) for op in self.order_parameters)

        # TODO: decide on which metrics to log
        self.weight_log = None
        self.state_log = None

    def _init_weights(self, prior_weights):
        """
        Effects
        ------
        Initialize weights with uniform values or use prior if defined.
        The ith weight siginifies the relative importance of the ith
        order parameter. Each step through the environment the weights
        are updated to encourage sampling of folded conformations.
        """
        if prior_weights:
            return np.array(prior_weights)

        # Uniform initialization
        weights = np.empty(self.num_order_params)
        weights.fill(1. / self.num_order_params)

        return weights

    def _reward(self, outlier, state):
        # Calculate mean and variance of each order parameter for the new state
        for op in self.order_parameters:
            op_val = op(state)
            self.stats[op] = Stats(np.mean(op_val), np.std(op_val))

        # Computes reward for a single outlier given state dependent statistics
        reward = lambda weight, op : weight * abs(op(outlier) - self.stats[op].mean) / self.stats[op].stdev

        # Computes sum of rewards over all order parameters for a particular outlier
        return sum(starmap(reward, zip(self.order_parameters, self.weights)))


    def _cumulative_reward(self, outliers, state):
        return sum(starmap(self._reward, zip(outliers, repeat(state))))

    def _optimize_weights(self):
        # Run maximization routine on the _cumulative_reward
        # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
        #  https://github.com/braceal/REAP-ReinforcementLearningBasedAdaptiveSampling/blob/master/MDSimulation/Src/RL/RLSim.py

    def find_best_structures(self, structures, min_samples):
        """
        Chooses protein structures to spawn new simulations
        based on least poplated clusters and the reward
        function of each cluster.
        """

        # 1. Add incomming structures to state
        self.state.append(structures)

        # 2. Cluster the state into a set of L clusters using OPTICS
        outlier_inds, labels = optics_clustering(self.state.data, min_samples)

        # 3. Identify subset of clusters which contain the least
        #    number of data points
        uniques, counts = np.unique(labels)
        # uniques: [-1, 1, 2, 3, 4, 5, ..., num_clusters]
        # counts:  [5, 10, 11, 15, 17]
        least_sampled = counts[np.argpartition(counts, kth=self.least_samples)][:self.least_samples + 1]

        # argpartition sorts the kth element with [... k ...] the lower half of the array having all values smaller than k
        # might not need to sort the k smallest values since we only need the associated clusters but not the size


        # 4. Compute reward for each structure in the subset of clusters

        # 5. Maximize reward using scipy to optimize weights

        # 6. Using updated weights choose new structures which give
        #    the greatest reward




        action = None
        return action