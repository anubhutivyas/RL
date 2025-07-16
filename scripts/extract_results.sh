# Usage: ./scripts/extract_results.sh results/Qwen2.5-7B-grpo-scp-*_summary.txt"

for file in "$@"; do
    if [ -f "$file" ]; then
        basename_file=$(basename "$file")        
        echo "$basename_file"
        
        # Extract steps from model_name lines
        steps=$(grep "model_name='hf_step_" "$file" | sed -n "s/.*model_name='hf_step_\([0-9]*\)'.*/\1/p" | tr '\n' ' ')
        
        # Extract scores and multiply by 100
        scores=$(grep "^score=" "$file" | sed -n "s/^score=\([0-9]*\.[0-9]*\).*/\1/p" | awk '{printf "%.2f ", $1*100}')
        
        # Convert to array format
        steps_array="[$(echo $steps | sed 's/ $//' | sed 's/ /, /g')]"
        scores_array="[$(echo $scores | sed 's/ $//' | sed 's/ /, /g')]"
        
        echo "steps: $steps_array"
        echo "scores: $scores_array"
        echo
    fi
done