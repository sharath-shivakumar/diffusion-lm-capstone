import json
import os

def split_midpoint(text):
    words = text.strip().split()
    if len(words) < 3:
        return None
    mid_idx = len(words) // 2
    left = " ".join(words[:mid_idx])
    mid = words[mid_idx]
    right = " ".join(words[mid_idx + 1:])
    return {
        "left_text": left,
        "mid_text": mid,
        "right_text": right
    }

def process_file(input_path, output_path, label):
    print(f"Processing {label}: {input_path} ? {output_path}")
    count = 0
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            if "||" not in line:
                continue
            _, nl = line.strip().split("||", 1)
            result = split_midpoint(nl)
            if result:
                json.dump(result, outfile)
                outfile.write("\n")
                count += 1
    print(f"? Processed {count} {label} samples into {output_path}\n")

def main():
    input_dir = "/home/exouser/Capstone/Diffusion-LM-main/datasets/e2e_data"
    output_dir = "/home/exouser/Capstone/Diffusion-LM-main/datasets/e2e_processed"
    os.makedirs(output_dir, exist_ok=True)

    file_map = {
        "train": ("src1_train.txt", "train_gen_spans.json"),
        "valid": ("src1_valid.txt", "valid_gen_spans.json"),
        "test":  ("src1_test.txt", "test.json"),
    }

    for label, (in_file, out_file) in file_map.items():
        input_path = os.path.join(input_dir, in_file)
        output_path = os.path.join(output_dir, out_file)
        process_file(input_path, output_path, label)

if __name__ == "__main__":
    main()