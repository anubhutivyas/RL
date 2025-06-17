# %%
import json
import numpy as np

filenames = [
    "nano_deepscaler_evals/nano_deepscaler_L1_data_context_20000.jsonl_bkp", 
    "nano_deepscaler_evals/nano_deepscaler_L1_data_context_20000_A.jsonl", 
    "nano_deepscaler_evals/nano_deepscaler_L1_data_context_20000_B.jsonl", 
    "nano_deepscaler_evals/nano_deepscaler_L1_data_context_20000_C.jsonl"]

N = len(filenames)

files = {}
# read each one of the files line by line, json decoding each one.  store each file json array in a dictionary with the name of the file as the dataset_key
for filename in filenames:
    with open(filename, "r") as f:
        files[filename] = [json.loads(line) for line in f]


problems = {}
for filename in filenames:
    for record in files[filename]:
        problem = record["original_problem"]
        if problem not in problems:
            problems[problem] = []
        problems[problem].append(record)

problem_stats = []
for problem in problems:
    num_correct = sum([record["correctness"]["content"]=="Environment: correct" for record in problems[problem]])
    num_incorrect = sum([record["correctness"]["content"]=="Environment: incorrect" for record in problems[problem]])
    num_total = len(problems[problem])
    sum_len_correct = sum([record["response_length"] for record in problems[problem] if record["correctness"]["content"]=="Environment: correct"])
    sum_len_incorrect = sum([record["response_length"] for record in problems[problem] if record["correctness"]["content"]=="Environment: incorrect"])
    av_len_correct = sum_len_correct / num_correct if num_correct > 0 else -1
    av_len_incorrect = sum_len_incorrect / num_incorrect if num_incorrect > 0 else -1
    problem_stats.append({
        "original_problem": problem,
        "num_correct": num_correct, 
        "num_total": num_total, 
        "av_len_correct": av_len_correct,
        "av_len_incorrect": av_len_incorrect
    })


with open("nano_deepscaler_evals/combined_deepscaler_evals.json", "w") as f:
    for problem_stat in problem_stats:
        f.write(json.dumps(problem_stat) + "\n")