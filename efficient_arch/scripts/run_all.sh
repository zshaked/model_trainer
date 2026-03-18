#!/bin/bash
# Run all three models with identical settings.
# Usage: bash scripts/run_all.sh [size] [dataset] [total_tokens]
#
# Examples:
#   bash scripts/run_all.sh small roneneldan/TinyStories 100000000  # Quick test
#   bash scripts/run_all.sh medium roneneldan/TinyStories 500000000 # Full run
#   bash scripts/run_all.sh medium openwebtext 1000000000           # Big run

SIZE=${1:-small}
DATASET=${2:-roneneldan/TinyStories}
TOKENS=${3:-100000000}
SEED=42

echo "============================================================"
echo "Architecture Comparison Experiment"
echo "  Size:    $SIZE"
echo "  Dataset: $DATASET"
echo "  Tokens:  $TOKENS"
echo "  Seed:    $SEED"
echo "============================================================"

# Run each model sequentially (same GPU, same conditions)
for MODEL in transformer ssm predictive_ssm; do
    echo ""
    echo "============================================================"
    echo "Training: $MODEL"
    echo "============================================================"
    python train.py \
        --model "$MODEL" \
        --size "$SIZE" \
        --dataset "$DATASET" \
        --total-tokens "$TOKENS" \
        --seed "$SEED" \
        --pred-loss-weight 0.05
done

echo ""
echo "============================================================"
echo "Comparing results..."
echo "============================================================"
python scripts/plot_curves.py runs/*_${SIZE}_seed${SEED}

echo ""
echo "Done! Check learning_curves.png for the comparison plot."
