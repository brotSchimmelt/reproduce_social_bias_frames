#!/bin/bash

start_time=$(date +%s)

arguments=("gpt2-xl-1701-1" "gpt2-xl-1701-2" "gpt2-xl-1701-5")

for arg in "${arguments[@]}"; do
    echo "Running inference.py with model: $arg"
    python inference.py --model_name "$arg" --max_length 128
done

echo "All inference runs completed."

end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$((elapsed % 60))
echo "Total execution time: $(printf "%02d:%02d:%02d" $hours $minutes $seconds)"
