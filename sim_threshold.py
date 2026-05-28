"""
Simulation 1 — Basic Threshold Model  (Step 1 + Step 2 of tech.tex)

Each paper is evaluated by two hypothetical reviewers: one with the
high-prestige prior and one with the low-prestige prior.  The same
content signal D is shown to both, so any difference in recommendation
is driven purely by the prestige prior.

Key question: with identical true innovation rates (TRUE_THETA = 0.5),
does prestige alone cause a systematic recommendation gap?
"""

import numpy as np
from params import (TRUE_THETA, THETA_HIGH, THETA_LOW,
                    MU_H1, MU_H0, SIGMA_FULL, TAU, N_PAPERS)
from utils import gaussian_bayes_update


def run(seed=None):
    """
    Run the threshold simulation.

    Returns
    -------
    list of dict, each with keys:
        prestige   : 'high' or 'low'
        H          : true innovation label (0 or 1)
        D          : observed content signal
        posterior  : P(H=1 | D, C)
        recommend  : 1 if posterior > TAU, else 0
    """
    rng = np.random.default_rng(seed)
    results = []

    for _ in range(N_PAPERS // 2):
        H = int(rng.random() < TRUE_THETA)
        # Content signal drawn from the paper's true quality, independent of prestige
        D = rng.normal(MU_H1 if H == 1 else MU_H0, SIGMA_FULL)

        for prestige, prior in [('high', THETA_HIGH), ('low', THETA_LOW)]:
            posterior = gaussian_bayes_update(prior, D, MU_H1, MU_H0, SIGMA_FULL)
            results.append({
                'prestige':  prestige,
                'H':         H,
                'D':         D,
                'posterior': posterior,
                'recommend': int(posterior > TAU),
            })

    return results


if __name__ == '__main__':
    from utils import rec_rate
    results = run(seed=42)

    print("=" * 55)
    print("Simulation 1: Basic Threshold Model")
    print("=" * 55)
    print(f"True innovation rate θ = {TRUE_THETA}  (identical for both groups)")
    print(f"High-prestige prior: {THETA_HIGH:.2f}   Low-prestige prior: {THETA_LOW:.2f}")
    print(f"Content signal: D | H ~ N(mu_H, {SIGMA_FULL}),  "
          f"mu_H1={MU_H1}, mu_H0={MU_H0}")
    print(f"Decision threshold τ = {TAU}")
    print()
    print(f"{'':22s} {'High':>8s} {'Low':>8s} {'Gap':>8s}")
    print("-" * 50)
    for label, h_val in [('All papers', None),
                          ('Truly innovative (H=1)', 1),
                          ('Ordinary (H=0)', 0)]:
        hv = rec_rate(results, 'high', h_val)
        lv = rec_rate(results, 'low',  h_val)
        print(f"Rec. rate ({label:<20s})  {hv:>7.3f}  {lv:>7.3f}  {hv-lv:>+7.3f}")
