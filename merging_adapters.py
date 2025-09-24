from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = "deepseek-ai/deepseek-coder-v2-lite-base"
dora_path = "./deepseek-dora-finetuned-small"  # put path of the last checkpoint

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)

print("Loading DoRA adapters...")
model = PeftModel.from_pretrained(model, dora_path)

print("Merging adapters...")
model = model.merge_and_unload()

save_path = "./deepseek-coder-v2-lite-base-dora-merged"
print(f"Saving merged model to {save_path}")
model.save_pretrained(save_path)
