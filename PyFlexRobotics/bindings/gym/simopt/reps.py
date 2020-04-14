#!/usr/bin/env python
""" Relative Entropy Policy Search (REPS) implementation.
J. Peters, K. Mulling, and Y. Altun. Relative entropy policy search. In AAAI, 2010.
 """
import numpy as np
import scipy as sp
from numpy import matlib as mb
from scipy.optimize import minimize


class Reps():

    def __init__(self, kl_threshold=1.0, covariance_damping=0.0,
                 min_temperature=0.001):
        self._kl_threshold = kl_threshold
        self._covariance_damping = covariance_damping
        self._min_temperature = min_temperature

    def learn(self, samples, costs, mean_old, cov_old):
        mean_new = np.zeros(mean_old.shape)
        cov_new = np.zeros(cov_old.shape)
        inv_cov_new = np.zeros(cov_old.shape)
        chol_cov_new = np.zeros(cov_old.shape)

        # Normalize costs.
        min_cost = np.min(costs)
        max_cost = np.max(costs)
        costs = (costs - min_cost) / (max_cost - min_cost + 0.00001)

        # Perform optimization of the temperature eta to ensure staying within the KL-bound.
        res = minimize(self.KL_dual, 1.0, bounds=((self._min_temperature,
                                                   None),), args=(self._kl_threshold, costs))
        eta = res.x

        # Compute probabilities of each sample.
        exp_cost = np.exp(-costs / eta)
        prob = exp_cost / np.sum(exp_cost)

        # Update policy mean with weighted max-likelihood.
        assert (prob.shape[0] == len(samples))
        mean_new = np.sum(mb.repmat(prob.reshape(prob.shape[0], 1), 1, len(samples[0])) * np.vstack(samples),
                          axis=0)

        # Update policy covariance with weighted max-likelihood.
        for i in range(len(samples)):
            mean_diff = samples[i] - mean_new
            mean_diff = np.reshape(mean_diff, (len(mean_diff), 1))
            cov_new += prob[i,] * np.dot(mean_diff, mean_diff.T)

        # If covariance damping is enabled, compute covariance as multiple
        # of the old covariance. The multiplier is first fitted using
        # max-likelihood and then taken to the power (1/covariance_damping).
        if (self._covariance_damping is not None
                and self._covariance_damping > 0.0):
            mult = np.trace(np.dot(sp.linalg.inv(cov_old),
                                   cov_new)) / len(cov_old)
            mult = np.power(mult, 1 / self._covariance_damping)
            cov_new = mult * cov_old

        # print ('rank(cov_new) = ', np.linalg.matrix_rank(cov_new))

        # Compute covariance inverse and cholesky decomposition.
        #        inv_cov_new = sp.linalg.inv(cov_new)
        #        chol_cov_new = sp.linalg.cholesky(cov_new)

        #        return mean_new, cov_new, inv_cov_new, chol_cov_new
        return mean_new, cov_new

    def KL_dual(self, eta, kl_threshold, costs):
        """
        Dual function for optimizing the temperature eta according to the given
        KL-divergence constraint.

        Args:
            eta: Temperature that has to be optimized.
            kl_threshold: Max. KL-divergence constraint.
            costs: Roll-out costs.
        Returns:
            Value of the dual function.
        """
        return eta * kl_threshold + eta * np.log((1.0 / len(costs)) *
                                                 np.sum(np.exp(-costs / eta)))

if __name__ == '__main__':
    curr_mean = np.asarray([0.0, 0.0])
    curr_cov = np.diag(np.asarray([1.0, 1.0]))
    target = np.asarray([10.0, 10.0])

    reps_optim = Reps(kl_threshold=1.0,
                      covariance_damping=2.0,
                      min_temperature=0.001)

    for i in range(100):
        samples = np.random.multivariate_normal(curr_mean, curr_cov, 50)
        costs = np.sum(np.abs(samples - target), axis=1)

        curr_mean, curr_cov = reps_optim.learn(samples, costs, curr_mean, curr_cov)

        print ("ITER " + str(i) + " meancost " + str(np.mean(costs)) + " mean " + str(curr_mean))


