import argparse
import re
import os
import sys
import contextlib
from transformers import AutoTokenizer

from nemo_rl.algorithms.utils import get_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_files", type=str, nargs="+")
    return parser.parse_args()


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def main():
    args = parse_args()
    
    if len(args.log_files) > 1:
        steps = [int(re.search(r"step_(\d+)", file).group(1)) for file in args.log_files]
        args.log_files = [file for _, file in sorted(zip(steps, args.log_files))]
    
    avg_input_lengths = []
    avg_output_lengths = []
    for log_file in args.log_files:
        with open(log_file, "r") as fin:
            section = None
            config_lines = []
            input_lines = []
            output_lines = []
            input_lengths = []
            output_lengths = []
            tokenizer = None
            
            for line in fin:
                if "Final config" in line:
                    section = "config"
                    continue
                elif "chat template" in line:
                    section = None
                    config = eval("".join(config_lines))
                    with suppress_stdout_stderr():
                        tokenizer = get_tokenizer(config["tokenizer"])
                elif "<<<<<<<<<<<<<<< inputs <<<<<<<<<<<<<<<" in line:
                    section = "input"
                    continue
                elif section == "input" and "======================================" in line:
                    section = "output"
                    continue
                elif ">>>>>>>>>>>>>>> outputs >>>>>>>>>>>>>>>" in line:
                    section = None
                    # remove the last \n
                    input = "".join(input_lines)[:-1]
                    output = "".join(output_lines)[:-1]
                    input_length = len(tokenizer.encode(input))
                    output_length = len(tokenizer.encode(output))
                    input_lengths.append(input_length)
                    output_lengths.append(output_length)
                    input_lines = []
                    output_lines = []
                if section == "config":
                    config_lines.append(line)
                if section == "input":
                    input_lines.append(line)
                elif section == "output":
                    output_lines.append(line)

        print(log_file)
        avg_input_length = sum(input_lengths) / len(input_lengths)
        avg_output_length = sum(output_lengths) / len(output_lengths)
        avg_input_lengths.append(avg_input_length)
        avg_output_lengths.append(avg_output_length)
        print(f"average input length={avg_input_length:.1f}")
        print(f"average output length={avg_output_length:.1f}")
        print()

    avg_input_lengths = ", ".join([f"{length:.1f}" for length in avg_input_lengths])
    avg_output_lengths = ", ".join([f"{length:.1f}" for length in avg_output_lengths])
    print(f"average input length array: [{avg_input_lengths}]")
    print(f"average output length array: [{avg_output_lengths}]")


if __name__ == "__main__":
    main()