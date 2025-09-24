# https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/Evaluation/LeetCode/vllm_inference.py

import json
import random
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def build_prompt(instruction):
    """Match your fine-tuning format."""
    return f"""### Instruction:
{instruction.strip()}
### Response:
"""

def generate_responses(llm, tokenizer, instructions, max_new_tokens=256):
    """Generate batched responses with vLLM."""
    prompts = [build_prompt(inst) for inst in instructions]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_new_tokens,
        stop=["### Instruction:"],  # optional safeguard
    )

    outputs = llm.generate(prompts, sampling_params)
    results = []
    for i, out in enumerate(outputs):
        generated = out.outputs[0].text.strip()

        # cleanup like in your LoRA script
        for token in ["<|EOT|>", "<|end|>", "<|endoftext|>"]:
            if token in generated:
                generated = generated.split(token)[0].strip()

        results.append(generated)
    return results

def evaluate_test_set(model_path, test_file, output_file, num_examples=None):
    print(f"Loading test data: {test_file}")
    with open(test_file, "r") as f:
        test_data = [json.loads(line) for line in f]

    if num_examples:
        test_data = test_data[:num_examples]

    print(f"Evaluating {len(test_data)} examples")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(model=model_path, gpu_memory_utilization=0.9, trust_remote_code=True)

    instructions = [item["instruction"] for item in test_data]
    expected_outputs = [item["output"] for item in test_data]

    generated_outputs = generate_responses(llm, tokenizer, instructions)

    results = []
    for i, (inst, expected, generated) in enumerate(zip(instructions, expected_outputs, generated_outputs)):
        result = {
            "example_id": i + 1,
            "instruction": inst,
            "expected_output": expected,
            "generated_output": generated,
            "expected_length": len(expected.split()),
            "generated_length": len(generated.split()),
            "status": "success"
        }
        results.append(result)

    print(f"Saving results to: {output_file}")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate merged DoRA model with vLLM")
    parser.add_argument("--model_path", default="./deepseek-coder-v2-lite-base-dora-merged", help="Path to merged DoRA model")
    parser.add_argument("--test_file", default="./finetune_data/test_data.json", help="Test data file")
    parser.add_argument("--output_file", default="./test_results.json", help="Output results file")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of examples to test (None for all)")
    args = parser.parse_args()

    results = evaluate_test_set(args.model_path, args.test_file, args.output_file, args.num_examples)

    successful = sum(1 for r in results if r["status"] == "success")
    print(f"\nEvaluation complete! Results saved to {args.output_file}")
    print(f"Successfully processed: {successful}/{len(results)} examples")

if __name__ == "__main__":
    main()
