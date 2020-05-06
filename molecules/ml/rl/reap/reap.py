import numpy as np
from scipy.optimize import minimize
from molecules.ml.unsupervised.cluster import optics_clustering

# TODO: could we also reward structures by computing the latent embedding of
#       the folded state and then attempting to minimize the distance to it?

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
        self.mean = np.array([])
        self.stdev = np.array([])

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

    def update(self, labels):
        """
        Parameters
        ----------
        labels: np.ndarray
            contains cluster label of each state in self.structures

        Effects
        -------
        Computes mean,stdev for each cluster in self.structures
        given the cluster labels.

        """
        means, stdevs = [], []

        for label in np.unique(labels):
            cluster = self.structures[labels == label]
            means.append(np.mean(cluster, axis=0))
            stdevs.append(np.std(cluster, axis=0))

        self.mean = np.mean(means, axis=0)
        self.stdev = np.std(stdevs, axis=0)


def topk(a, k):
    """
    Parameters
    ----------
    a : np.ndarray
        array of dim (N,)

    k : int
        specifies which element to partition upon

    Returns
    -------
    np.ndarray of length k containing indices of input array a
    coresponding to the k smallest values in a.

    """
    return np.argpartition(a, k)[:k]

def least_sampling_set(cluster_labels, k):
    """
    Parameters
    ----------
    cluster_labels : np.ndarray
        array of dim (N,)

    k : int
        specifies which element to partition upon

    Returns
    -------
    np.ndarray of length k containing the cluster labels
    of the k smallest clusters.

    """
    labels, counts = np.unique(cluster_labels, return_counts=True)
    if k >= len(labels):
        return labels
    return labels[topk(counts, k)]


class REAP:
    """
    Environement: landscape that is to be explored
    State:        set of all discovered points (latent embeddings)
                  on the landscape
    Action:       choosing protein structures to run more simulations on
    """
    def __init__(self, num_order_params, least_samples=5,
                 num_spawns=10, prior_weights=None):

        # CVAE latent dimension
        self.num_order_params = num_order_params

        # Number of clusters to examine when computing rewards
        self.least_samples = least_samples

        # Number of simulations to spawn each action
        self.num_spawns = num_spawns

        # Same size as the number of order parameters. The ith weight
        # represents the importance of the ith order parameter.
        self.weights = self._init_weights(prior_weights)

        self.state = State()

        # TODO: decide on which metrics to log
        # self.weight_log = None
        # self.state_log = None

        # TODO: pass in path to hdf5 file storing state

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

    def _reward_structure(self, structure, weights):
        return np.dot(weights, np.abs((structure - self.state.mean)) / self.state.stdev)

    def _reward_cluster(self, cluster, weights):
        # TODO: should each outlier be treated as it's own cluster?
        # Currently compressing each cluster to its mean
        return _reward_cluster(np.mean(cluster, axis=0), weights)

    def _cumulative_reward(self, clusters, weights):
        return sum(_reward_cluster(cluster, weights) for cluster in clusters)

    def _optimize_weights(self, clusters):
        # Run maximization routine on the _cumulative_reward
        # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
        #  https://github.com/braceal/REAP-ReinforcementLearningBasedAdaptiveSampling/blob/master/MDSimulation/Src/RL/RLSim.py

        def fun(x):
            return -1. * self._cumulative_reward(clusters, x)

        delta = 0.1
        constraints = ({'type': 'eq',
                        'fun' : lambda x: np.array([np.sum(x)-1])}, # weights sum to one
                       {'type': 'ineq',
                        'fun' : lambda x: np.array([np.min(x)])}, # greater than zero
                       {'type': 'ineq',
                        'fun' : lambda x: np.array([-np.abs(x[0]-x0[0])+delta])}, # greater than zero
                       {'type': 'ineq',
                        'fun' : lambda x: np.array([-np.abs(x[1]-x0[1])+delta])}, # greater than zero
                       {'type': 'ineq',
                        'fun' : lambda x: np.array([-np.abs(x[2]-x0[2])+delta])}, # greater than zero
                       {'type': 'ineq',
                        'fun' : lambda x: np.array([-np.abs(x[3]-x0[3])+delta])}) # greater than zero

        result = minimize(fun=fun, x0=self.weights,
                          constraints=constraints, method='SLSQP')

        if result.success:
            self.weights = result.weights

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
        _, labels = optics_clustering(self.state.structures, params)

        # Update state with cluster stats
        self.state.update(labels)

        # 3. Identify subset of clusters which contain the least
        #    number of data points
        best_labels = least_sampling_set(labels, self.least_samples)
        clusters = np.array([self.state.structures[labels == label] for label in best_labels])

        # 4. Maximize reward using scipy to optimize weights
        self._optimize_weights(clusters)

        # 5. Using updated weights choose new structures which give
        #    the greatest reward

        # Compute array of rewards for each structure
        # Negative for efficiently finding the best k
        rewards = np.array([-1. * self._reward_structure(structure, self.weights)
                            for structure in structures])
        # Unsorted array of indices coresponding to the best structures
        best_structure_inds = topk(rewards, self.num_spawns)

        # TODO: should possible spawns be a subset of the incomming structures,
        #       or should we be able to spawn and previous simulation structure
        #       in the state. (Currently the first is implemented)

        # TODO: consider whether we always want the same number of simulations spawned
        #       or if there should be a schedule (large at start, less towards end).


        return best_structure_inds
