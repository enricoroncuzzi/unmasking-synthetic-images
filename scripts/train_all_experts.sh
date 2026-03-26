#!/usr/bin/env bash

EXPERTS=("sd15" "sd21" "sdxl_base" "sd35" "flux")

for expert in "${EXPERTS[@]}"; do
    echo "=========================================="
    echo "Starting training: $expert"
    echo "=========================================="
    python training/train_expert.py \
        expert=$expert \
        training.batch_size=16 \
        data.num_workers=4 \
        2>&1 | tee ${expert}.log
    echo "Finished: $expert"
done

echo "All experts trained."