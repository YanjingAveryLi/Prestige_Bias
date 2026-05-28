"""
Core Bayesian utility functions shared across all simulations.
"""

import numpy as np
from scipy.stats import norm


def gaussian_bayes_update(prior, d_obs, mu_h1, mu_h0, sigma):
    """
    Bayesian posterior update with a Gaussian likelihood.

    Given a prior P(H=1) and an observed continuous content signal d_obs,
    returns the posterior P(H=1 | d_obs) using:
        P(H=1 | d) ∝ N(d; mu_h1, sigma) * prior

    Parameters
    ----------
    prior  : float  Prior probability P(H=1 | C)
    d_obs  : float  Observed content signal
    mu_h1  : float  Expected signal mean if H=1
    mu_h0  : float  Expected signal mean if H=0
    sigma  : float  Signal noise (standard deviation)

    Returns
    -------
    float  Posterior P(H=1 | d_obs, C)
    """
    lh1 = norm.pdf(d_obs, mu_h1, sigma)
    lh0 = norm.pdf(d_obs, mu_h0, sigma)
    num = lh1 * prior
    denom = num + lh0 * (1.0 - prior)
    return num / denom if denom > 0 else prior


def rec_rate(results, prestige, h=None):
    """Recommendation rate for a given prestige group, optionally filtered by true H."""
    subset = [r for r in results
              if r['prestige'] == prestige and (h is None or r['H'] == h)]
    return np.mean([r['recommend'] for r in subset]) if subset else 0.0


def deep_rate(results, prestige, h=None):
    """Deep-read trigger rate for a given prestige group, optionally filtered by true H."""
    subset = [r for r in results
              if r['prestige'] == prestige and (h is None or r['H'] == h)]
    return np.mean([r['deep_read'] for r in subset]) if subset else 0.0


def innovative_rec_by_depth(results, prestige):
    """
    For truly innovative papers (H=1) of a given prestige group, return the
    recommendation rate separately for papers that received deep reading vs
    those that received only shallow reading.

    Returns
    -------
    (deep_rec_rate, shallow_rec_rate) : tuple of float
    """
    deep   = [r for r in results if r['prestige'] == prestige and r['H'] == 1 and r['deep_read']]
    shallow= [r for r in results if r['prestige'] == prestige and r['H'] == 1 and not r['deep_read']]
    rd = np.mean([r['recommend'] for r in deep])    if deep    else 0.0
    rs = np.mean([r['recommend'] for r in shallow]) if shallow else 0.0
    return rd, rs
