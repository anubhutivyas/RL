#! /bin/bash

MODEL=$1
OUTDIR=detailed_evals

# create OUTDIR if it doesn't exist
mkdir -p $OUTDIR

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model>"
    exit 1
fi

echo "MODEL: $MODEL"
SHORT_MODEL_NAME=`python -c "s='$MODEL' if '$MODEL'[-1]!='/' else '$MODEL'[:-1];  print('-'.join(s.split('/')[-2:]))"`
echo "SHORT_MODEL_NAME: $SHORT_MODEL_NAME"

# create a directory $OUTDIR/SHORT_MODEL_NAME if it doesn't exist
mkdir -p $OUTDIR/$SHORT_MODEL_NAME

# if the model name has "OpenThinker" in it, then set the system and user prompt file to one thing, otherwise set them to the nano option
if [[ "$MODEL" == *"OpenThinker"* ]]; then
    echo "Using OpenThinker prompts"
    SYSTEM_PROMPT_FILE="examples/prompts/cot_openthinker_system.txt"
    USER_PROMPT_FILE="examples/prompts/cot_openthinker_user.txt"
else
    echo "Using nano prompts"
    SYSTEM_PROMPT_FILE="examples/prompts/cot_nano_system.txt"
    USER_PROMPT_FILE="examples/prompts/cot_nano_user.txt"
fi

# iterate over datasets AIME2024 AIME2025 GPQA-D and MATH500
for dataset in HuggingFaceH4/aime_2024 MathArena/aime_2025 zwhe99/gpqa_diamond_mc HuggingFaceH4/MATH-500; do
    echo "--------------------------------"
    echo DATASET: $dataset

    # set SHORT_MODEL_NAME to 
    
    # number of repetitions as well as dataset keys depends on the dataset
    if [ $dataset = "HuggingFaceH4/MATH-500" ]; then
        dataset_key="test"
        problem_key="problem"
        solution_key="answer"
        gpus_per_node=5
        num_prompts_per_step=500
        num_repetitions=1
        short_dataset_name="MATH-500"
    elif [ $dataset = "HuggingFaceH4/aime_2024" ]; then
        dataset_key="train"
        problem_key="problem"
        solution_key="answer"
        gpus_per_node=5
        num_prompts_per_step=30
        num_repetitions=5
        short_dataset_name="AIME2024"
    elif [ $dataset = "MathArena/aime_2025" ]; then
        dataset_key="train"
        problem_key="problem"
        solution_key="answer"
        num_repetitions=5
        gpus_per_node=5
        num_prompts_per_step=30
        num_repetitions=10
        short_dataset_name="AIME2025"
    elif [ $dataset = "zwhe99/gpqa_diamond_mc" ]; then
        dataset_key="test"
        problem_key="problem"
        solution_key="solution"
        num_prompts_per_step=198
        gpus_per_node=6
        num_repetitions=5
        short_dataset_name="GPQA-D"
    fi
    echo "dataset_key: $dataset_key"
    echo "problem_key: $problem_key"
    echo "solution_key: $solution_key"

    mkdir -p $OUTDIR/$SHORT_MODEL_NAME/$short_dataset_name

    # repeat num_repetitions times
    for i in $(seq 1 $num_repetitions); do
        echo "Running test $i for dataset $dataset"
        uv run python examples/run_eval_with_planted_thinking.py generation.model_name=$MODEL data.shuffle_seed=$RANDOM data.dataset_name=$dataset data.dataset_key=$dataset_key data.problem_key=$problem_key data.solution_key=$solution_key generation.vllm_cfg.max_model_len=24000 generation.num_prompts_per_step=$num_prompts_per_step cluster.gpus_per_node=$gpus_per_node debug.outfile=$OUTDIR/$SHORT_MODEL_NAME/$short_dataset_name/$i.jsonl data.prompt_file=$USER_PROMPT_FILE data.system_prompt_file=$SYSTEM_PROMPT_FILE --config tests/configs/eval_nano.yaml   2>& 1 | grep -E "score=|Mean length"
    done
done

