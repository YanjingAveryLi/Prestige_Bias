"""
Simulation 2 — Cumulative Advantage  (Step 3 of tech.tex)

The Beta prior parameters (alpha, b) update dynamically after each
evaluation round, so the perceived innovation rate θ̂_C drifts over time
based on accumulated recommendation history.

Because early recommendations are biased by the prestige prior, the
perceived innovation rate for the low-prestige group is systematically
driven below the true value — even though both groups have the same
true innovation rate (TRUE_THETA = 0.5).
"""

import numpy as np
from params import (TRUE_THETA, ALPHA_HIGH, B_HIGH, ALPHA_LOW, B_LOW,
                    MU_H1, MU_H0, SIGMA_FULL, TAU,
                    N_ROUNDS, PAPERS_PER_ROUND)
from utils import gaussian_bayes_update


def run(seed=None):
    """
    Run the cumulative advantage simulation.

    Returns
    -------
    s_high        : list[int]   Cumulative recommendations for high-prestige group
    s_low         : list[int]   Cumulative recommendations for low-prestige group
    theta_h_track : list[float] Perceived innovation rate θ̂_{C+} over rounds
    theta_l_track : list[float] Perceived innovation rate θ̂_{C-} over rounds
    """
    rng = np.random.default_rng(seed)

    # Beta parameters — updated after every recommendation
    a_h, b_h = float(ALPHA_HIGH), float(B_HIGH)
    a_l, b_l = float(ALPHA_LOW),  float(B_LOW)

    s_high, s_low = [0], [0]
    theta_h_track = [a_h / (a_h + b_h)]
    theta_l_track = [a_l / (a_l + b_l)]

    for _ in range(N_ROUNDS):
        rh = rl = 0
        for _ in range(PAPERS_PER_ROUND // 2):
            H = int(rng.random() < TRUE_THETA)
            D = rng.normal(MU_H1 if H == 1 else MU_H0, SIGMA_FULL)

            for prestige in ['high', 'low']:
                # Use current posterior mean as the prior for this round
                prior = a_h / (a_h + b_h) if prestige == 'high' else a_l / (a_l + b_l)
                rec = int(gaussian_bayes_update(prior, D, MU_H1, MU_H0, SIGMA_FULL) > TAU)

                # Update Beta parameters: recommendation counts as a "success"
                if prestige == 'high':
                    a_h += rec;  b_h += (1 - rec);  rh += rec
                else:
                    a_l += rec;  b_l += (1 - rec);  rl += rec

        s_high.append(s_high[-1] + rh)
        s_low.append(s_low[-1]   + rl)
        theta_h_track.append(a_h / (a_h + b_h))
        theta_l_track.append(a_l / (a_l + b_l))

    return s_high, s_low, theta_h_track, theta_l_track


if __name__ == '__main__':
    s_high, s_low, th, tl = run(seed=42)
    total = N_ROUNDS * PAPERS_PER_ROUND / 2

    print("=" * 55)
    print("Simulation 2: Cumulative Advantage")
    print("=" * 55)
    print(f"Rounds: {N_ROUNDS},  Papers per round per group: {PAPERS_PER_ROUND // 2}")
    print(f"True innovation rate θ = {TRUE_THETA} for both groups")
    print()
    print(f"High-prestige cumulative recommendations: {s_high[-1]}  "
          f"({s_high[-1]/total*100:.1f}%)")
    print(f"Low-prestige  cumulative recommendations: {s_low[-1]}   "
          f"({s_low[-1]/total*100:.1f}%)")
    print(f"Final perceived innovation rate — high: {th[-1]:.3f},  "
          f"low: {tl[-1]:.3f}  (true: {TRUE_THETA})")
