import re
import json
import argparse
from tqdm import tqdm
from math_verify import parse


# this script is originally written for the SCP dataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="filtered.jsonl")
    parser.add_argument("--output", type=str, default="verifiable.jsonl")
    args = parser.parse_args()

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        valid = 0
        total = 0
        for line in tqdm(fin):
            data = json.loads(line)
            answer = data["expected_answer"]
            parsed = parse(r"\boxed{" + answer + "}")
            total += 1
            if r"\text" in answer:
                # textual answers, discard
                # note this also discards all physical quantities like 100\text{F}
                continue
            if len(parsed) < 2:
                # no symbol or expression found, discard
                continue
            # might be either long latex formulas or chemical phrases
            words = re.split(r"[()\[\]{}+\-*/ ]+", answer)
            # latex formulas usually don't contain long words except commands containing "\"
            has_long_word = any("\\" not in word and len(word) >= 10 for word in words)
            if has_long_word:
                # probably chemical phrases, discard
                continue

            fout.write(line)
            valid += 1

    ratio = valid / total
    print(f"{total} lines processed, {valid} ({ratio:.2%}) verifiable")