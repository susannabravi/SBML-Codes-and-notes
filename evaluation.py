import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import os

def load_finetuned_model(base_model_path: str, lora_adapter_path: str, device: str = "cuda"):
    """Load the base model and apply LoRA adapters."""
    
    print(f"Loading base model: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left"  # For generation
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    print(f"Loading LoRA adapters: {lora_adapter_path}")
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    # Merge adapters for faster inference (optional)
    model = model.merge_and_unload()
    
    print("Model loaded successfully!")
    return model, tokenizer

def build_instruction_prompt(instruction: str):
    """Build the same prompt format used during training."""
    return '''### Instruction:
{}
### Response:
'''.format(instruction.strip())

def generate_response(model, tokenizer, instruction: str, max_length: int = 512, 
                     temperature: float = 0.7, top_p: float = 0.9, device: str = "cuda"):
    """Generate response for a given instruction."""
    
    # Build prompt
    prompt = build_instruction_prompt(instruction)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|EOT|>"),
            repetition_penalty=1.1
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after "### Response:")
    response_start = full_response.find("### Response:") + len("### Response:")
    generated_text = full_response[response_start:].strip()
    
    # Remove EOT token if present
    if "<|EOT|>" in generated_text:
        generated_text = generated_text.split("<|EOT|>")[0].strip()
    
    return generated_text

def evaluate_on_dataset(model, tokenizer, data_path: str, device: str = "cuda", 
                       max_samples: int = None, output_file: str = None):
    """Evaluate model on validation/test dataset."""
    
    print(f"Loading dataset: {data_path}")
    
    # Load dataset
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            if data_path.endswith('.jsonl') or 'jsonl' in data_path:
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
    else:
        raise ValueError("Unsupported file format. Use JSON or JSONL.")
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"Evaluating on {len(data)} samples...")
    
    results = []
    
    for i, item in enumerate(tqdm(data, desc="Generating responses")):
        instruction = item['instruction']
        expected_output = item['output']
        
        # Generate response
        try:
            generated_output = generate_response(model, tokenizer, instruction, device=device)
            
            result = {
                'sample_id': i,
                'instruction': instruction,
                'expected_output': expected_output,
                'generated_output': generated_output,
                'instruction_length': len(instruction),
                'expected_length': len(expected_output),
                'generated_length': len(generated_output)
            }
            
            results.append(result)
            
            # Print first few examples
            if i < 3:
                print(f"\n{'='*50}")
                print(f"Sample {i+1}:")
                print(f"Instruction: {instruction[:200]}...")
                print(f"Expected: {expected_output[:200]}...")
                print(f"Generated: {generated_output[:200]}...")
                print(f"{'='*50}\n")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Save results if output file specified
    if output_file:
        print(f"Saving results to: {output_file}")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def calculate_simple_metrics(results: List[Dict]):
    """Calculate simple evaluation metrics."""
    
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    # Basic statistics
    total_samples = len(results)
    avg_instruction_length = np.mean([r['instruction_length'] for r in results])
    avg_expected_length = np.mean([r['expected_length'] for r in results])
    avg_generated_length = np.mean([r['generated_length'] for r in results])
    
    print(f"Total samples evaluated: {total_samples}")
    print(f"Average instruction length: {avg_instruction_length:.1f} chars")
    print(f"Average expected response length: {avg_expected_length:.1f} chars")
    print(f"Average generated response length: {avg_generated_length:.1f} chars")
    
    # Length ratio (generated vs expected)
    length_ratios = [r['generated_length'] / max(r['expected_length'], 1) for r in results]
    avg_length_ratio = np.mean(length_ratios)
    print(f"Average length ratio (gen/exp): {avg_length_ratio:.2f}")
    
    # Simple word overlap (basic similarity measure)
    word_overlaps = []
    for r in results:
        expected_words = set(r['expected_output'].lower().split())
        generated_words = set(r['generated_output'].lower().split())
        
        if len(expected_words) > 0:
            overlap = len(expected_words.intersection(generated_words)) / len(expected_words)
            word_overlaps.append(overlap)
    
    if word_overlaps:
        avg_word_overlap = np.mean(word_overlaps)
        print(f"Average word overlap: {avg_word_overlap:.3f}")
    
    print("="*60)

def show_detailed_examples(model, tokenizer, data_path: str, device: str = "cuda", num_examples: int = 5):
    """Show detailed examples with full input/output comparison."""
    
    print(f"Loading dataset: {data_path}")
    
    # Load dataset
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            lines = f.readlines()
            data = [json.loads(line) for line in lines]
    else:
        raise ValueError("Unsupported file format. Use JSON.")
    
    print(f"\nShowing {num_examples} detailed examples from the dataset:")
    print("="*100)
    
    for i in range(min(num_examples, len(data))):
        item = data[i]
        instruction = item['instruction']
        expected_output = item['output']
        
        print(f"\nEXAMPLE {i+1}:")
        print("-" * 80)
        print(f"INSTRUCTION:\n{instruction}")
        print(f"\nEXPECTED OUTPUT:\n{expected_output}")
        
        # Generate response
        try:
            print(f"\nGENERATING RESPONSE...")
            generated_output = generate_response(model, tokenizer, instruction, device=device)
            print(f"\nGENERATED OUTPUT:\n{generated_output}")
            
            # Simple comparison
            expected_words = len(expected_output.split())
            generated_words = len(generated_output.split())
            print(f"\nCOMPARISON:")
            print(f"Expected length: {expected_words} words ({len(expected_output)} chars)")
            print(f"Generated length: {generated_words} words ({len(generated_output)} chars)")
            
            # Word overlap
            expected_word_set = set(expected_output.lower().split())
            generated_word_set = set(generated_output.lower().split())
            overlap = len(expected_word_set.intersection(generated_word_set))
            overlap_ratio = overlap / len(expected_word_set) if expected_word_set else 0
            print(f"Word overlap: {overlap}/{len(expected_word_set)} ({overlap_ratio:.2%})")
            
        except Exception as e:
            print(f"ERROR GENERATING RESPONSE: {e}")
        
        print("="*100)
        
        # Pause between examples for readability
        if i < num_examples - 1:
            input("Press Enter to see next example...")

def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned DeepSeek model")
    parser.add_argument("--base_model", type=str, default="deepseek-ai/deepseek-coder-1.3b-base",
                       help="Base model path")
    parser.add_argument("--lora_path", type=str, default="./output_small",
                       help="Path to LoRA adapters")
    parser.add_argument("--test_data", type=str, default="./finetune_data/test_data.json",
                       help="Test dataset path")
    parser.add_argument("--val_data", type=str, default="./finetune_data/val_data.json",
                       help="Validation dataset path")
    parser.add_argument("--num_examples", type=int, default=5,
                       help="Number of examples to show in detail")
    parser.add_argument("--max_samples", type=int, default=50,
                       help="Maximum samples to evaluate for metrics (None for all)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--show_examples", action="store_true", default=True,
                       help="Show detailed examples")
    parser.add_argument("--run_full_eval", action="store_true",
                       help="Run full evaluation with metrics")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading fine-tuned model...")
    model, tokenizer = load_finetuned_model(args.base_model, args.lora_path, args.device)
    
    # Show detailed examples first
    if args.show_examples:
        if os.path.exists(args.test_data):
            print(f"\n{'='*60}")
            print("DETAILED EXAMPLES FROM TEST SET")
            print(f"{'='*60}")
            show_detailed_examples(model, tokenizer, args.test_data, args.device, args.num_examples)
        
        elif os.path.exists(args.val_data):
            print(f"\n{'='*60}")
            print("DETAILED EXAMPLES FROM VALIDATION SET")
            print(f"{'='*60}")
            show_detailed_examples(model, tokenizer, args.val_data, args.device, args.num_examples)
    
    # Run full evaluation if requested
    if args.run_full_eval:
        # Test on validation set
        if os.path.exists(args.val_data):
            print(f"\nEvaluating on validation set...")
            val_results = evaluate_on_dataset(
                model, tokenizer, args.val_data, args.device, 
                args.max_samples, f"{args.output_dir}/val_results.json"
            )
            calculate_simple_metrics(val_results)
        
        # Test on test set  
        if os.path.exists(args.test_data):
            print(f"\nEvaluating on test set...")
            test_results = evaluate_on_dataset(
                model, tokenizer, args.test_data, args.device,
                args.max_samples, f"{args.output_dir}/test_results.json"
            )
            calculate_simple_metrics(test_results)

if __name__ == "__main__":
    main()