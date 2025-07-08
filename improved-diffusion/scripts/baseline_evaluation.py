import torch
from torch.utils.data import DataLoader
import pandas as pd
import json
import wandb
from evaluation_metrics import DiffusionLMEvaluator
from datasets.e2e_preprocessor import E2EPreprocessor
from datasets.e2e_dataset import E2EDataset

class BaselineEvaluationPipeline:
    def __init__(self, model_path, vocab_path, test_data_path, device='cuda'):
        self.device = device
        self.evaluator = DiffusionLMEvaluator(device)
        
        # Load model and preprocessor
        self.model = self.load_model(model_path)
        self.preprocessor = self.load_preprocessor(vocab_path)
        
        # Load test data
        self.test_dataset = E2EDataset(test_data_path, self.preprocessor, 'test')
        self.test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=False)
        
        # Load reference texts for BLEU computation
        test_df = pd.read_csv(test_data_path)
        self.reference_texts = test_df['ref'].tolist()
        
    def load_model(self, model_path):
        """Load the trained diffusion model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        # Assuming you have a DiffusionLM class defined
        # model = DiffusionLM(**model_config)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # return model.to(self.device)
        pass  # Placeholder - implement based on your model class
    
    def load_preprocessor(self, vocab_path):
        """Load the preprocessor with vocabulary"""
        import pickle
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        
        preprocessor = E2EPreprocessor(vocab_size=len(vocab), seq_len=64)
        preprocessor.vocab = vocab
        preprocessor.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        preprocessor.idx_to_word = {idx: word for word, idx in preprocessor.word_to_idx.items()}
        
        return preprocessor
    
    def generate_samples(self, num_samples=1000):
        """Generate samples from the model for evaluation"""
        self.model.eval()
        generated_texts = []
        
        with torch.no_grad():
            sample_count = 0
            for batch in self.test_loader:
                if sample_count >= num_samples:
                    break
                    
                batch = batch.to(self.device)
                
                # Generate samples (implement based on your model's sampling method)
                # generated_tokens = self.model.sample(batch)
                
                # Convert tokens back to text
                # for tokens in generated_tokens:
                #     text = self.tokens_to_text(tokens)
                #     generated_texts.append(text)
                #     sample_count += 1
                
                pass  # Placeholder - implement based on your model
        
        return generated_texts[:num_samples]
    
    def tokens_to_text(self, tokens):
        """Convert token indices back to text"""
        words = []
        for token_idx in tokens:
            if token_idx == self.preprocessor.word_to_idx['<PAD>']:
                break
            if token_idx in [self.preprocessor.word_to_idx['<BOS>'], self.preprocessor.word_to_idx['<EOS>']]:
                continue
            words.append(self.preprocessor.idx_to_word[token_idx.item()])
        return ' '.join(words)
    
    def create_control_constraints(self, num_samples):
        """Create control constraints for evaluation"""
        # Example constraints for E2E dataset
        constraints = []
        
        for i in range(num_samples):
            # Semantic control constraint
            constraint = {
                'type': 'semantic',
                'required_terms': ['restaurant', 'food'],  # Basic terms for E2E
            }
            constraints.append(constraint)
        
        return constraints
    
    def run_full_evaluation(self, num_samples=1000):
        """Run complete evaluation pipeline"""
        print("Starting baseline evaluation...")
        
        # Generate samples
        print("Generating samples...")
        generated_texts = self.generate_samples(num_samples)
        reference_texts = self.reference_texts[:len(generated_texts)]
        
        # Create control constraints
        control_constraints = self.create_control_constraints(len(generated_texts))
        
        # Compute all metrics
        print("Computing metrics...")
        
        # 1. Control Success Rate
        control_success = self.evaluator.compute_control_success_rate(
            generated_texts, control_constraints
        )
        
        # 2. LM-Score (Perplexity)
        lm_score = self.evaluator.compute_lm_score(generated_texts)
        
        # 3. BLEU Scores
        bleu_scores = self.evaluator.compute_bleu_scores(generated_texts, reference_texts)
        
        # 4. Decoding Speed
        speed_metrics = self.evaluator.measure_decoding_speed(
            self.model, next(iter(self.test_loader))
        )
        
        # 5. Quality/Controllability Trade-off
        tradeoff_metrics = self.evaluator.compute_quality_controllability_tradeoff(
            generated_texts, reference_texts, control_constraints
        )
        
        # Compile results
        results = {
            'control_success_rate': control_success,
            'lm_score': lm_score,
            'bleu_scores': bleu_scores,
            'decoding_speed': speed_metrics,
            'tradeoff_metrics': tradeoff_metrics,
            'num_samples': len(generated_texts)
        }
        
        return results
    
    def log_results(self, results, experiment_name="baseline_evaluation"):
        """Log results to wandb and save locally"""
        
        # Log to wandb
        wandb.init(project="diffusion-lm-cosine", name=experiment_name)
        wandb.log(results)
        
        # Save locally
        with open(f'{experiment_name}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("BASELINE EVALUATION RESULTS")
        print("="*50)
        print(f"Control Success Rate: {results['control_success_rate']:.4f}")
        print(f"LM-Score (Perplexity): {results['lm_score']:.4f}")
        print(f"BLEU-4: {results['bleu_scores']['BLEU-4']:.4f}")
        print(f"Decoding Speed: {results['decoding_speed']['tokens_per_second']:.2f} tokens/sec")
        print(f"Quality/Control Trade-off: {results['tradeoff_metrics']['composite_score']:.4f}")
        print("="*50)