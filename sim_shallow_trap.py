"""
Simulation 3 — Shallow-Reading Trap  (Step 2, depth-of-processing, of tech.tex)

Evaluators first do a shallow read (noisy content signal).  The resulting
posterior falls into one of three zones:

  Reject zone    posterior < TAU - DEEP_ZONE  →  reject without deep reading
  Uncertain zone |posterior - TAU| < DEEP_ZONE  →  trigger deep reading
  Accept zone    posterior > TAU + DEEP_ZONE  →  accept without deep reading

Key finding: low-prestige innovative papers (H=1) disproportionately land
in the reject zone after shallow reading — not because they fail to trigger
deep reading, but because the low prior drags the posterior below the
uncertain zone entirely, so they are rejected before ever reaching it.
"""

import numpy as np
from params import (TRUE_THETA, THETA_HIGH, THETA_LOW,
                    MU_H1, MU_H0, SIGMA_FULL, SIGMA_SHALLOW,
                    TAU, DEEP_ZONE, N_PAPERS)
from utils import gaussian_bayes_update


def run(seed=None):
    """
    Run the shallow-reading trap simulation.

    Returns
    -------
    list of dict, each with keys:
        prestige     : 'high' or 'low'
        H            : true innovation label (0 or 1)
        post_shallow : posterior after shallow read
        final_post   : posterior after deep read (if triggered), else post_shallow
        deep_read    : bool — whether deep reading was triggered
        recommend    : 1 if final_post > TAU, else 0
    """
    rng = np.random.default_rng(seed)
    results = []

    for _ in range(N_PAPERS // 2):
        H = int(rng.random() < TRUE_THETA)

        for prestige, prior in [('high', THETA_HIGH), ('low', THETA_LOW)]:
            # Stage 1: shallow read — noisy content signal
            D_s = rng.normal(MU_H1 if H == 1 else MU_H0, SIGMA_SHALLOW)
            post_s = gaussian_bayes_update(prior, D_s, MU_H1, MU_H0, SIGMA_SHALLOW)

            # Stage 2: deep read — only if posterior is in the uncertain zone
            if abs(post_s - TAU) < DEEP_ZONE:
                D_d = rng.normal(MU_H1 if H == 1 else MU_H0, SIGMA_FULL)
                # Use shallow posterior as the new prior for the deep-read update
                final_post = gaussian_bayes_update(post_s, D_d, MU_H1, MU_H0, SIGMA_FULL)
                deep_read = True
            else:
                final_post = post_s
                deep_read  = False

            results.append({
                'prestige':     prestige,
                'H':            H,
                'post_shallow': post_s,
                'final_post':   final_post,
                'deep_read':    deep_read,
                'recommend':    int(final_post > TAU),
            })

    return results


if __name__ == '__main__':
    from utils import deep_rate, innovative_rec_by_depth
    results = run(seed=42)

    print("=" * 55)
    print("Simulation 3: Shallow-Reading Trap")
    print("=" * 55)
    print(f"Shallow σ={SIGMA_SHALLOW},  Deep σ={SIGMA_FULL},  "
          f"Uncertain zone ±{DEEP_ZONE} around τ={TAU}")
    print()
    print(f"{'':28s} {'High':>8s} {'Low':>8s}")
    print("-" * 48)
    for label, h_val in [('All papers', None),
                          ('Truly innovative (H=1)', 1),
                          ('Ordinary (H=0)', 0)]:
        hv = deep_rate(results, 'high', h_val)
        lv = deep_rate(results, 'low',  h_val)
        print(f"Deep-read trigger rate ({label:<18s})  {hv:>7.3f}  {lv:>7.3f}")

    hd_r, hs_r = innovative_rec_by_depth(results, 'high')
    ld_r, ls_r = innovative_rec_by_depth(results, 'low')
    print()
    print("Recommendation rate for innovative papers (H=1):")
    print(f"  High prestige — deep reading: {hd_r:.3f}  /  shallow only: {hs_r:.3f}")
    print(f"  Low  prestige — deep reading: {ld_r:.3f}  /  shallow only: {ls_r:.3f}")
