#!/usr/bin/env bash
# Train all 4 MoE gating strategies sequentially.
#
# Mirrors train_all_experts.sh from Phase 2.
# Run from the project root:
#
#   bash scripts/train_all_moe.sh
#
# Each strategy logs to experiments/mlruns under experiment "moe_training".
# Checkpoints saved to checkpoints/moe/{strategy}/best-*.ckpt
#
# To run in background and keep logs:
#   nohup bash scripts/train_all_moe.sh > logs/train_all_moe.log 2>&1 &

set -e  # exit immediately on any error

STRATEGIES=("logit" "embedding" "image" "attention")

echo "========================================"
echo " MoE gating network training — Phase 3"
echo " Strategies: ${STRATEGIES[*]}"
echo "========================================"
echo ""

for strategy in "${STRATEGIES[@]}"; do
    echo "----------------------------------------"
    echo " Starting: moe=${strategy}"
    echo " $(date '+%Y-%m-%d %H:%M:%S')"
    echo "----------------------------------------"

    python training/train_moe.py moe="${strategy}"

    echo ""
    echo " Finished: moe=${strategy} — $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
done

echo "========================================"
echo " All 4 strategies complete."
echo "========================================"