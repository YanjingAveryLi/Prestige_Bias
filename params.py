"""
Shared parameters for all prestige-bias simulations.
Corresponds to the three-step computational framework in tech.tex.
"""

import numpy as np

# Reproducibility
RANDOM_SEED = 42

# --- Step 1: Hierarchical Bayesian prior (Beta-Bernoulli) ---
# Prior means derived from Beta(alpha, b)
ALPHA_HIGH, B_HIGH = 8.0, 2.0   # High-prestige prior: mean = 0.80
ALPHA_LOW,  B_LOW  = 2.0, 8.0   # Low-prestige prior:  mean = 0.20
THETA_HIGH = ALPHA_HIGH / (ALPHA_HIGH + B_HIGH)   # 0.80
THETA_LOW  = ALPHA_LOW  / (ALPHA_LOW  + B_LOW)    # 0.20

# --- Content signal (Gaussian likelihood) ---
# D | H ~ N(MU_H, SIGMA), where MU_H1 > MU_H0
MU_H1 = 0.70    # Mean content signal for truly innovative papers (H=1)
MU_H0 = 0.30    # Mean content signal for ordinary papers (H=0)
SIGMA_FULL    = 0.20   # Signal noise for full / deep reading
SIGMA_SHALLOW = 0.38   # Signal noise for shallow reading (higher noise)

# --- Step 2: Decision threshold ---
TAU = 0.50       # Recommendation threshold: recommend iff posterior > TAU

# Uncertain zone for triggering deep reading: |post_shallow - TAU| < DEEP_ZONE
DEEP_ZONE = 0.15

# --- Ground truth (null hypothesis) ---
# Both groups have the SAME true innovation rate.
# All observed inequality is driven purely by the prestige prior.
TRUE_THETA = 0.50

# --- Simulation sizes ---
N_PAPERS         = 800    # Total papers per simulation (split evenly between groups)
N_ROUNDS         = 70     # Rounds for the cumulative advantage simulation
PAPERS_PER_ROUND = 20     # Papers evaluated each round (split evenly)
