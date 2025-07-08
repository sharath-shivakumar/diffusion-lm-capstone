import json
import argparse
from evaluation_metrics import DiffusionLMEvaluator

def load_generated_texts(json_path):
    generated_texts = []
    with open(json_path, "r") as f:
        for line in f:
            try:
                batch = json.loads(line)
                if isinstance(batch, list):
                    for text in batch:
                        cleaned = text.strip()
                        if cleaned:
                            generated_texts.append(cleaned)
            except json.JSONDecodeError:
                continue
    return generated_texts
    
def load_reference_texts(ref_path):
    refs = []
    with open(ref_path, "r") as f:
        for line in f:
            if "||" in line:
                parts = line.strip().split("||")
                if len(parts) == 2:
                    refs.append(parts[1].strip())
    return refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypothesis", required=True, help="Path to generated text (.json)")
    parser.add_argument("--reference", required=True, help="Path to reference text file (e.g., src1_test.txt)")
    args = parser.parse_args()

    gen_texts = load_generated_texts(args.hypothesis)
    ref_texts = load_reference_texts(args.reference)

    evaluator = DiffusionLMEvaluator()
    bleu_scores = evaluator.compute_bleu_scores(gen_texts, ref_texts)
    perplexity = evaluator.compute_lm_score(gen_texts)

    print("\nEvaluation Results:")
    for k, v in bleu_scores.items():
        print(f"{k}: {v * 100:.2f}")
    print(f"Perplexity: {perplexity:.2f}")


if __name__ == "__main__":
    main()
