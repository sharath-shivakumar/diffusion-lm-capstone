import torch
import torch.nn.functional as F
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
from collections import Counter
import re

class DiffusionLMEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        # Load GPT-2 for perplexity calculation (LM-Score)
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        self.smoothing = SmoothingFunction().method1
        
    def compute_control_success_rate(self, generated_texts, control_constraints):
        """
        Compute control success rate based on constraint satisfaction
        Args:
            generated_texts: List of generated text strings
            control_constraints: List of constraint dictionaries
        Returns:
            success_rate: Float between 0 and 1
        """
        successful_controls = 0
        total_constraints = len(control_constraints)
        
        for text, constraint in zip(generated_texts, control_constraints):
            if self._check_constraint_satisfaction(text, constraint):
                successful_controls += 1
                
        return successful_controls / total_constraints if total_constraints > 0 else 0.0
    
    def _check_constraint_satisfaction(self, text, constraint):
        """Check if generated text satisfies the given constraint"""
        constraint_type = constraint.get('type', '')
        
        if constraint_type == 'semantic':
            # Check if required keywords/phrases are present
            required_terms = constraint.get('required_terms', [])
            text_lower = text.lower()
            return all(term.lower() in text_lower for term in required_terms)
            
        elif constraint_type == 'length':
            # Check length constraints
            target_length = constraint.get('target_length', 0)
            tolerance = constraint.get('tolerance', 5)
            actual_length = len(text.split())
            return abs(actual_length - target_length) <= tolerance
            
        elif constraint_type == 'pos':
            # POS tag constraint checking (simplified)
            target_pos_pattern = constraint.get('pos_pattern', [])
            # This would require actual POS tagging - simplified for now
            return True  # Placeholder
            
        elif constraint_type == 'syntax':
            # Syntax tree constraint checking (simplified)
            return True  # Placeholder - would need actual parsing
            
        return False
    
    def compute_lm_score(self, texts, batch_size=16):
        """
        Compute LM-Score (perplexity) using GPT-2
        Lower scores indicate better fluency
        """
        self.gpt2_model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize batch
                encoded = self.gpt2_tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Compute loss
                outputs = self.gpt2_model(**encoded, labels=encoded['input_ids'])
                loss = outputs.loss
                
                # Count non-padding tokens
                non_pad_tokens = (encoded['attention_mask'] == 1).sum().item()
                
                total_loss += loss.item() * non_pad_tokens
                total_tokens += non_pad_tokens
        
        # Return perplexity
        avg_loss = total_loss / total_tokens
        return np.exp(avg_loss)
    
    def compute_bleu_scores(self, generated_texts, reference_texts):
        """
        Compute BLEU scores for text similarity
        Args:
            generated_texts: List of generated text strings
            reference_texts: List of reference text strings
        Returns:
            bleu_scores: Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4
        """
        bleu_1_scores = []
        bleu_2_scores = []
        bleu_3_scores = []
        bleu_4_scores = []
        
        for gen_text, ref_text in zip(generated_texts, reference_texts):
            gen_tokens = gen_text.lower().split()
            ref_tokens = [ref_text.lower().split()]  # BLEU expects list of references
            
            # Compute different BLEU scores
            bleu_1 = sentence_bleu(ref_tokens, gen_tokens, weights=(1, 0, 0, 0), smoothing_function=self.smoothing)
            bleu_2 = sentence_bleu(ref_tokens, gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothing)
            bleu_3 = sentence_bleu(ref_tokens, gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smoothing)
            bleu_4 = sentence_bleu(ref_tokens, gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing)
            
            bleu_1_scores.append(bleu_1)
            bleu_2_scores.append(bleu_2)
            bleu_3_scores.append(bleu_3)
            bleu_4_scores.append(bleu_4)
        
        return {
            'BLEU-1': np.mean(bleu_1_scores),
            'BLEU-2': np.mean(bleu_2_scores),
            'BLEU-3': np.mean(bleu_3_scores),
            'BLEU-4': np.mean(bleu_4_scores)
        }
    
    def measure_decoding_speed(self, model, test_inputs, num_runs=5):
        """
        Measure decoding speed (tokens per second)
        Args:
            model: The diffusion model
            test_inputs: Test input data
            num_runs: Number of runs for averaging
        Returns:
            tokens_per_second: Average decoding speed
        """
        model.eval()
        total_time = 0
        total_tokens = 0
        
        with torch.no_grad():
            for run in range(num_runs):
                start_time = time.time()
                
                # Generate samples
                generated_samples = model.sample(test_inputs)
                
                end_time = time.time()
                
                # Count tokens generated
                batch_size, seq_len = generated_samples.shape
                tokens_generated = batch_size * seq_len
                
                total_time += (end_time - start_time)
                total_tokens += tokens_generated
        
        avg_time = total_time / num_runs
        tokens_per_second = total_tokens / total_time
        
        return {
            'tokens_per_second': tokens_per_second,
            'avg_generation_time': avg_time,
            'total_tokens': total_tokens
        }
    
    def compute_quality_controllability_tradeoff(self, generated_texts, reference_texts, control_constraints):
        """
        Compute the trade-off between quality and controllability
        Returns a composite score balancing both aspects
        """
        # Quality metrics
        bleu_scores = self.compute_bleu_scores(generated_texts, reference_texts)
        lm_score = self.compute_lm_score(generated_texts)
        
        # Controllability metric
        control_success = self.compute_control_success_rate(generated_texts, control_constraints)
        
        # Normalize and combine (higher is better)
        quality_score = bleu_scores['BLEU-4'] * (1.0 / max(lm_score, 1.0))  # Higher BLEU, lower perplexity
        controllability_score = control_success
        
        # Weighted combination
        composite_score = 0.6 * quality_score + 0.4 * controllability_score
        
        return {
            'quality_score': quality_score,
            'controllability_score': controllability_score,
            'composite_score': composite_score,
            'bleu_4': bleu_scores['BLEU-4'],
            'lm_score': lm_score,
            'control_success_rate': control_success
        }