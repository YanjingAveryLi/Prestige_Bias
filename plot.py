"""
Visualization — six-panel figure for the prestige-bias simulation.

Panels
------
1. Posterior distributions (same content signal, different prestige prior)
2. Recommendation rates by group and true quality
3. Dynamic evolution of perceived innovation rate θ̂_C
4. Cumulative recognition divergence S_t
5. Shallow-reading trap: deep-read trigger rates
6. Innovative papers (H=1): recommendation rate for deep vs shallow reading
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from params import TRUE_THETA, TAU, THETA_HIGH, THETA_LOW
from utils import rec_rate, deep_rate, innovative_rec_by_depth

BLUE = 'steelblue'
RED  = 'tomato'
W    = 0.35
OUT  = '/Users/1532016078qq.com/Desktop/PSYC 35880/final/paper/project/simulation_results.png'


def draw(threshold_results, s_high, s_low, theta_h_track, theta_l_track,
         deep_results, save_path=OUT):
    """
    Build and save the six-panel figure.

    Parameters
    ----------
    threshold_results : output of sim_threshold.run()
    s_high, s_low     : cumulative recommendation lists from sim_cumulative.run()
    theta_h_track,
    theta_l_track     : perceived innovation rate tracks from sim_cumulative.run()
    deep_results      : output of sim_shallow_trap.run()
    save_path         : file path for the saved PNG
    """
    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    # ------------------------------------------------------------------
    # Panel 1: Posterior distributions
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ph = [r['posterior'] for r in threshold_results if r['prestige'] == 'high']
    pl = [r['posterior'] for r in threshold_results if r['prestige'] == 'low']
    ax1.hist(ph, bins=40, alpha=0.65, color=BLUE, label='High Prestige', density=True)
    ax1.hist(pl, bins=40, alpha=0.65, color=RED,  label='Low Prestige',  density=True)
    ax1.axvline(TAU, color='k', ls='--', lw=1.5, label=f'Threshold τ={TAU}')
    ax1.set_xlabel('Posterior P(H=1 | D, C)')
    ax1.set_ylabel('Density')
    ax1.set_title('Panel 1  Posterior Distributions\n'
                  '(Same content signal, different prestige)')
    ax1.legend(fontsize=9)

    # ------------------------------------------------------------------
    # Panel 2: Recommendation rates
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    cats = ['All Papers', 'Truly Innovative\n(H=1)', 'Ordinary\n(H=0)']
    hr = [rec_rate(threshold_results, 'high'),
          rec_rate(threshold_results, 'high', 1),
          rec_rate(threshold_results, 'high', 0)]
    lr = [rec_rate(threshold_results, 'low'),
          rec_rate(threshold_results, 'low',  1),
          rec_rate(threshold_results, 'low',  0)]
    x = np.arange(3)
    ax2.bar(x - W/2, hr, W, color=BLUE, alpha=0.85, label='High Prestige')
    ax2.bar(x + W/2, lr, W, color=RED,  alpha=0.85, label='Low Prestige')
    ax2.axhline(TRUE_THETA, color='gray', ls=':', lw=1.2,
                label=f'True innovation rate {TRUE_THETA}')
    ax2.set_xticks(x); ax2.set_xticklabels(cats, fontsize=9)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel('Recommendation Rate')
    ax2.set_title('Panel 2  Recommendation Rates\n'
                  '(True innovation rate identical across groups)')
    ax2.legend(fontsize=9)
    for xi, (h, l) in enumerate(zip(hr, lr)):
        ax2.text(xi - W/2, h + 0.02, f'{h:.2f}', ha='center', fontsize=8, color=BLUE)
        ax2.text(xi + W/2, l + 0.02, f'{l:.2f}', ha='center', fontsize=8, color=RED)

    # ------------------------------------------------------------------
    # Panel 3: Dynamic evolution of θ̂_C
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    rng = range(len(theta_h_track))
    ax3.plot(rng, theta_h_track, color=BLUE, lw=2,
             label='High Prestige $\\hat{\\theta}_{C_+}$')
    ax3.plot(rng, theta_l_track, color=RED,  lw=2,
             label='Low Prestige $\\hat{\\theta}_{C_-}$')
    ax3.axhline(TRUE_THETA, color='gray', ls='--', lw=1.5,
                label=f'True value θ={TRUE_THETA}')
    ax3.set_xlabel('Evaluation Round')
    ax3.set_ylabel('$\\hat{\\theta}_C$')
    ax3.set_title('Panel 3  Evolution of Perceived Innovation Rate\n'
                  '(Prior lock-in diverges from ground truth)')
    ax3.legend(fontsize=9)

    # ------------------------------------------------------------------
    # Panel 4: Cumulative recognition divergence
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    rounds = range(len(s_high))
    ax4.plot(rounds, s_high, color=BLUE, lw=2, label='High Prestige')
    ax4.plot(rounds, s_low,  color=RED,  lw=2, label='Low Prestige')
    ax4.fill_between(rounds, s_high, s_low,
                     alpha=0.13, color='gray', label='Recognition gap')
    ax4.set_xlabel('Evaluation Round')
    ax4.set_ylabel('Cumulative Recommendations $S_t$')
    ax4.set_title('Panel 4  Cumulative Recognition Divergence\n'
                  '(Path dependence and snowball effect)')
    ax4.legend(fontsize=9)

    # ------------------------------------------------------------------
    # Panel 5: Deep-read trigger rates (shallow-reading trap)
    # ------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    cats2 = ['All Papers', 'Truly Innovative\n(H=1)', 'Ordinary\n(H=0)']
    hd = [deep_rate(deep_results, 'high'),
          deep_rate(deep_results, 'high', 1),
          deep_rate(deep_results, 'high', 0)]
    ld = [deep_rate(deep_results, 'low'),
          deep_rate(deep_results, 'low',  1),
          deep_rate(deep_results, 'low',  0)]
    x2 = np.arange(3)
    ax5.bar(x2 - W/2, hd, W, color=BLUE, alpha=0.85, label='High Prestige')
    ax5.bar(x2 + W/2, ld, W, color=RED,  alpha=0.85, label='Low Prestige')
    ax5.set_xticks(x2); ax5.set_xticklabels(cats2, fontsize=9)
    ax5.set_ylim(0, 0.70); ax5.set_ylabel('Deep-Read Trigger Rate')
    ax5.set_title('Panel 5  Shallow-Reading Trap\n'
                  'Deep-read trigger rate by group and true quality')
    ax5.legend(fontsize=9)
    for xi, (h, l) in enumerate(zip(hd, ld)):
        ax5.text(xi - W/2, h + 0.01, f'{h:.2f}', ha='center', fontsize=8, color=BLUE)
        ax5.text(xi + W/2, l + 0.01, f'{l:.2f}', ha='center', fontsize=8, color=RED)

    # ------------------------------------------------------------------
    # Panel 6: Innovative papers — deep vs shallow recommendation rate
    # ------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    hd_r, hs_r = innovative_rec_by_depth(deep_results, 'high')
    ld_r, ls_r = innovative_rec_by_depth(deep_results, 'low')
    labels6 = ['High\n(Deep)', 'High\n(Shallow)', 'Low\n(Deep)', 'Low\n(Shallow)']
    vals6   = [hd_r, hs_r, ld_r, ls_r]
    cols6   = [BLUE, 'lightsteelblue', RED, 'lightsalmon']
    bars = ax6.bar(labels6, vals6, color=cols6, alpha=0.92)
    ax6.set_ylim(0, 1.10); ax6.set_ylabel('Recommendation Rate')
    ax6.set_title('Panel 6  Innovative Papers (H=1)\n'
                  'Recommendation rate: deep vs shallow reading')
    for bar, val in zip(bars, vals6):
        ax6.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')

    fig.suptitle(
        'Computational Simulation of Prestige Bias  |  '
        'True innovation rate identical across groups (θ = 0.5)',
        fontsize=13, fontweight='bold'
    )

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure saved: {save_path}")
