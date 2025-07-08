import argparse
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baseline_evaluation import BaselineEvaluationPipeline

def main():
    parser = argparse.ArgumentParser(description='Run baseline evaluation for Diffusion-LM')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to vocabulary file')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for evaluation')
    parser.add_argument('--experiment_name', type=str, default='baseline_sqrt_schedule', help='Experiment name')
    
    args = parser.parse_args()
    
    # Initialize evaluation pipeline
    evaluator = BaselineEvaluationPipeline(
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        test_data_path=args.test_data,
        device=args.device
    )
    
    # Run evaluation
    results = evaluator.run_full_evaluation(num_samples=args.num_samples)
    
    # Log and save results
    evaluator.log_results(results, args.experiment_name)
    
    print(f"\nEvaluation completed! Results saved as {args.experiment_name}_results.json")

if __name__ == "__main__":
    main()