"""
Copy-paste this into Google Colab cells to run the full experiment.
Each section between # --- CELL --- markers is a separate Colab cell.
"""

# --- CELL 1: Setup ---
# !pip install torch transformers datasets matplotlib
# !git clone <your-repo-url> efficient_arch
# %cd efficient_arch

# --- CELL 2: Quick smoke test (5 min) ---
"""
# Run a tiny experiment first to make sure everything works
!python train.py --model transformer --size small --dataset roneneldan/TinyStories --total-tokens 5000000
!python train.py --model ssm --size small --dataset roneneldan/TinyStories --total-tokens 5000000
!python train.py --model predictive_ssm --size small --dataset roneneldan/TinyStories --total-tokens 5000000
!python scripts/plot_curves.py runs/*_small_seed42
"""

# --- CELL 3: Full experiment (2-4 hours on T4) ---
"""
# TinyStories, medium models, 500M tokens each
!python train.py --model transformer --size medium --dataset roneneldan/TinyStories --total-tokens 500000000
!python train.py --model ssm --size medium --dataset roneneldan/TinyStories --total-tokens 500000000
!python train.py --model predictive_ssm --size medium --dataset roneneldan/TinyStories --total-tokens 500000000 --pred-loss-weight 0.05
!python scripts/plot_curves.py runs/*_medium_seed42
"""

# --- CELL 4: Multiple seeds for significance (6-12 hours) ---
"""
for seed in [42, 123, 456]:
    for model in ['transformer', 'ssm', 'predictive_ssm']:
        !python train.py --model {model} --size medium --dataset roneneldan/TinyStories --total-tokens 500000000 --seed {seed}

# Plot all seeds
!python scripts/plot_curves.py runs/*_medium_*
"""
