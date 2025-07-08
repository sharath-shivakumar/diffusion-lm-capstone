import argparse
import ast
import json
from tqdm import tqdm

def flatten_dict_output(input_path, output_path):
    outputs = []

    with open(input_path, 'r') as infile:
        for line in tqdm(infile, desc="Processing lines"):
            try:
                parsed = ast.literal_eval(line.strip())  # use ast instead of json
                if isinstance(parsed, dict):
                    for v in parsed.values():
                        if isinstance(v, list):
                            outputs.extend(v)
                        else:
                            outputs.append(str(v))
            except Exception as e:
                print(f"Skipping line due to parsing error: {e}")

    with open(output_path, 'w') as outfile:
        json.dump(outputs, outfile, indent=2)

    print(f"? Wrote {len(outputs)} sentences to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flatten control task output dictionary into list of texts")
    parser.add_argument('--input', type=str, required=True, help='Path to input .json file')
    parser.add_argument('--output', type=str, required=True, help='Path to output .json file')
    args = parser.parse_args()

    flatten_dict_output(args.input, args.output)
