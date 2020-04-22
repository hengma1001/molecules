import numpy as np
from collections import namedtuple
from itertools import starmap, repeat

class REAP:
    def __init__(self, order_parameters, prior_weights=None):

        # TODO: may need to pass in more data for order params
        #       such as rmsd, etc.

        # List of functions to evaluate to compute each order parameter
        # given the data
        self.order_parameters = order_parameters

        # Same size as the number of order parameters. The ith weight
        # represents the importance of the ith order parameter.
        self.weights = self._init_weights(prior_weights)

        Stats = namedtuple('Stats', ['mean', 'stdev'])
        self.stats = dict((OP, Stats(0., 1.)) for OP in self.order_parameters)

        # TODO: decide on which metrics to log
        self.weight_log = None
        self.state_log = None

    def _init_weights(self, prior_weights):
        """
        Initialize weights with uniform values or use prior if defined.
        """
        if prior_weights:
            return prior_weights

        # Uniform Initialization
        shape = len(self.order_parameters)
        weights = np.empty(shape)
        weights.fill(1. / shape)

        return weights

    def _reward(self, outlier, state):
        # Calculate mean and variance of each order parameter for the new state
        for OP in self.order_parameters:
            op_val = OP(state)
            self.stats[OP] = Stats(np.mean(op_val), np.std(op_val))


        # Computes reward for a single outlier given state dependent statistics
        reward = lambda weight, OP : weight * abs(OP(outlier) - self.stats[OP].mean) / self.stats[OP].stdev

        # Computes sum of rewards over all order parameters for a particular outlier
        return sum(starmap(reward, zip(self.order_parameters, self.weights)))


    def _cumulative_reward(self, outliers, state):
        return sum(starmap(_reward, zip(outliers, repeat(state))))

    def _optimize_weights(self):
        # Run maximization routine on the _cumulative_reward
        # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
