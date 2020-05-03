import numpy as np
from collections import namedtuple
from itertools import starmap, repeat
from molecules.ml.unsupervised.cluster import optics_clustering

# TODO: use h5py
#       https://stackoverflow.com/questions/25655588/incremental-writes-to-hdf5-with-h5py?rq=1
#       https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py#47074545

# TODO: instead of storing the entire history to run optics
#       clustering on, we could store a representative sample
class State:
    def __init__(self):
        #State:    set of all discovered points (latent embeddings)
        #          on the landscape
        self.structures = np.array([])

    def save(self, fname):
        np.save(fname, self.structures)

    def load(self, fname):
        self.structures = np.load(fname)

    def append(self, structures):
        """
        Effects
        -------
        Appends new structures to state

        """
        self.structures = np.concatenate((self.structures, structures))


def least_sampling_sort(cluster_labels, k):
    # TODO: might not need to sort the k smallest values since we
    #       only need the associated clusters but not the size
    cluster_ids, counts = np.unique(cluster_labels, return_counts=True)
    if k >= len(cluster_ids):
        return cluster_ids[np.argsort(counts)]
    partition = np.argpartition(counts, k)[:k]
    return cluster_ids[partition[np.argsort(counts[partition])]]

def least_sampling_set(cluster_labels, k):
    cluster_ids, counts = np.unique(cluster_labels, return_counts=True)
    if k >= len(cluster_ids):
        return cluster_ids
    return cluster_ids[np.argpartition(counts, k)[:k]]


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
        pass

    def find_best_structures(self, structures, params):
        """
        Chooses protein structures to spawn new simulations
        based on least poplated clusters and the reward
        function of each cluster.

        Note: params should contain min_samples = 10
        """

        # 1. Add incomming structures to state
        self.state.append(structures)

        # 2. Cluster the state into a set of L clusters using OPTICS
        outlier_inds, labels = optics_clustering(self.state.structures, params)

        # 3. Identify subset of clusters which contain the least
        #    number of data points
        clusters = least_sampling_set(labels, self.least_samples)

        # 4. Compute reward for each structure in the subset of clusters

        # 5. Maximize reward using scipy to optimize weights

        # 6. Using updated weights choose new structures which give
        #    the greatest reward




        action = None
        return action