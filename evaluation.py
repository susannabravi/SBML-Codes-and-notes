import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

def load_model(base_model_path, lora_path):
    """Load base model + LoRA adapters."""
    print(f"Loading model: {base_model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()  # Merge for faster inference
    
    print("Model loaded successfully!")
    return model, tokenizer

def build_prompt(instruction):
    """Build the same prompt format used during training."""
    return f'''### Instruction:
{instruction.strip()}
### Response:
'''

def generate_response(model, tokenizer, instruction, max_new_tokens=256):
    """Generate response for given instruction."""
    prompt = build_prompt(instruction)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|EOT|>"),
        )
    
    # Decode and extract response
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = full_text.find("### Response:") + len("### Response:")
    response = full_text[response_start:].strip()
    
    # Remove EOT token if present
    if "<|EOT|>" in response:
        response = response.split("<|EOT|>")[0].strip()
    
    return response

def evaluate_test_set(model, tokenizer, test_file, output_file, num_examples=None):
    """Evaluate on test set and save results."""
    print(f"Loading test data: {test_file}")
    
    # Load test data
    with open(test_file, 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    if num_examples:
        test_data = test_data[:num_examples]
    
    print(f"Evaluating {len(test_data)} examples")
    
    results = []
    
    for i, item in enumerate(test_data):
        instruction = item['instruction']
        expected = item['output']
        
        print(f"Processing example {i+1}/{len(test_data)}")
        
        # Generate response
        generated = generate_response(model, tokenizer, instruction)
        
        # Store result
        result = {
            "example_id": i + 1,
            "instruction": instruction,
            "expected_output": expected,
            "generated_output": generated,
            "expected_length": len(expected.split()),
            "generated_length": len(generated.split())
        }
        
        results.append(result)
    
    # Save results
    print(f"Saving results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned model")
    parser.add_argument("--base_model", default="deepseek-ai/deepseek-coder-1.3b-base", 
                       help="Base model path")
    parser.add_argument("--lora_path", default="./output_small", 
                       help="LoRA adapters path")
    parser.add_argument("--test_file", default="./finetune_data/test_data.json", 
                       help="Test data file")
    parser.add_argument("--output_file", default="./test_results.json", 
                       help="Output results file")
    parser.add_argument("--num_examples", type=int, default=None, 
                       help="Number of examples to test (None for all)")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.base_model, args.lora_path)
    
    # Run evaluation
    results = evaluate_test_set(
        model, tokenizer, 
        args.test_file, 
        args.output_file, 
        args.num_examples
    )
    
    print(f"\nEvaluation complete! Results saved to {args.output_file}")

if __name__ == "__main__":
    main()