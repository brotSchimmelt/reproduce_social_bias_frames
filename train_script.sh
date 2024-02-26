#!/bin/bash

start_time=$(date +%s)

EPOCHS=(2 5)
SEEDS=(42)

# loop over epochs
for epoch in "${EPOCHS[@]}"; do
    # loop over seeds
    for seed in "${SEEDS[@]}"; do
        echo "Training with ${epoch} epochs and seed ${seed}..."
        python train.py --epochs $epoch --random_seed $seed
        echo "Training completed for ${epoch} epochs with seed ${seed}."
    done
done

echo "All training runs completed."

end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$((elapsed % 60))
echo "Total execution time: $(printf "%02d:%02d:%02d" $hours $minutes $seconds)"
