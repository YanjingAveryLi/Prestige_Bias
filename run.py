"""
Main entry point — runs all three simulations and generates the figure.

Usage
-----
    python run.py

Output
------
    simulation_results.png   Six-panel figure (also embedded in tech.pdf)
"""

from params import RANDOM_SEED, TRUE_THETA, N_ROUNDS, PAPERS_PER_ROUND

import sim_threshold
import sim_cumulative
import sim_shallow_trap
import plot
from utils import rec_rate, deep_rate, innovative_rec_by_depth


def main():
    print("Running simulations...")

    # --- Simulation 1: threshold model ---
    threshold_results = sim_threshold.run(seed=RANDOM_SEED)

    # --- Simulation 2: cumulative advantage ---
    s_high, s_low, theta_h_track, theta_l_track = sim_cumulative.run(seed=RANDOM_SEED)

    # --- Simulation 3: shallow-reading trap ---
    deep_results = sim_shallow_trap.run(seed=RANDOM_SEED)

    # --- Figure ---
    plot.draw(threshold_results, s_high, s_low,
              theta_h_track, theta_l_track, deep_results)

    # --- Numerical summary ---
    total_rounds = N_ROUNDS * PAPERS_PER_ROUND / 2

    print()
    print("=" * 55)
    print("Simulation 1: Basic Threshold Model")
    print("=" * 55)
    print(f"True innovation rate θ = {TRUE_THETA}  (identical for both groups)")
    print()
    print(f"{'':28s} {'High':>8s} {'Low':>8s} {'Gap':>8s}")
    print("-" * 56)
    for label, h_val in [('All papers',             None),
                          ('Truly innovative (H=1)', 1),
                          ('Ordinary (H=0)',          0)]:
        hv = rec_rate(threshold_results, 'high', h_val)
        lv = rec_rate(threshold_results, 'low',  h_val)
        print(f"Rec. rate ({label:<20s})  {hv:>7.3f}  {lv:>7.3f}  {hv-lv:>+7.3f}")

    print()
    print("=" * 55)
    print("Simulation 3: Shallow-Reading Trap")
    print("=" * 55)
    print()
    print(f"{'':28s} {'High':>8s} {'Low':>8s}")
    print("-" * 48)
    for label, h_val in [('All papers',             None),
                          ('Truly innovative (H=1)', 1),
                          ('Ordinary (H=0)',          0)]:
        hv = deep_rate(deep_results, 'high', h_val)
        lv = deep_rate(deep_results, 'low',  h_val)
        print(f"Deep-read trigger rate ({label:<18s})  {hv:>7.3f}  {lv:>7.3f}")
    hd_r, hs_r = innovative_rec_by_depth(deep_results, 'high')
    ld_r, ls_r = innovative_rec_by_depth(deep_results, 'low')
    print()
    print("Recommendation rate for innovative papers (H=1):")
    print(f"  High prestige — deep: {hd_r:.3f}  /  shallow only: {hs_r:.3f}")
    print(f"  Low  prestige — deep: {ld_r:.3f}  /  shallow only: {ls_r:.3f}")

    print()
    print("=" * 55)
    print("Simulation 2: Cumulative Advantage (final state)")
    print("=" * 55)
    print(f"High-prestige cumulative recommendations: {s_high[-1]}"
          f"  ({s_high[-1]/total_rounds*100:.1f}%)")
    print(f"Low-prestige  cumulative recommendations: {s_low[-1]}"
          f"  ({s_low[-1]/total_rounds*100:.1f}%)")
    print(f"Final perceived innovation rate — "
          f"high: {theta_h_track[-1]:.3f},  low: {theta_l_track[-1]:.3f}"
          f"  (true: {TRUE_THETA})")


if __name__ == '__main__':
    main()
