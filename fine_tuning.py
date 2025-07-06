# https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/finetune/finetune_deepseekcoder.py
import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.distributed
import transformers
from transformers import Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType


IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"

def build_instruction_prompt(instruction: str):
    """
    Build instruction prompt for base model fine-tuning.
    Using a simple instruction-response format that works well with base models.
    """
    return '''### Instruction:
{}
### Response:
'''.format(instruction.strip())

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="deepseek-ai/deepseek-coder-v2-lite-base")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,  # Reduced for memory efficiency
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

    # LoRA-optimized training parameters
    learning_rate: float = field(default=3e-4)  # Higher LR for LoRA
    warmup_ratio: float = field(default=0.03)
    weight_decay: float = field(default=0.0)
    # LoRA configuration
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer):
    sources = [
        build_instruction_prompt(instruction)
        for instruction in examples['instruction']
    ]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['output']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

# Added because I'm not using the instruct one
def setup_tokenizer(model_path: str, model_max_length: int):
    """Setup tokenizer for base model with proper special tokens."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    
    # Handle special tokens for base model
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Add EOT token if not present
    if EOT_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': [EOT_TOKEN]})
    
    return tokenizer

def setup_lora_model(model, training_args):
    """Setup LoRA configuration and apply to model."""
    
    # LoRA configuration targeting key attention and MLP layers
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
            "gate_proj", "up_proj", "down_proj"      # MLP layers
        ],
        bias="none",
        modules_to_save=["embed_tokens", "lm_head"]  # Save embedding layers
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if training_args.local_rank == 0:
        print('='*100)
        print(training_args)
        print(f"Fine-tuning DeepSeek Coder V2 Base with LoRA: {model_args.model_name_or_path}")
    
    # Setup tokenizer with special handling for base model (added because I'm not using the instruct)
    tokenizer = setup_tokenizer(model_args.model_name_or_path, training_args.model_max_length)

    if training_args.local_rank == 0:
        print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
        print("BOS Token:", tokenizer.bos_token, tokenizer.bos_token_id)
        print("EOS Token:", tokenizer.eos_token, tokenizer.eos_token_id)
        print("EOT Token:", tokenizer.convert_tokens_to_ids(EOT_TOKEN))
        print("Vocab size:", len(tokenizer))

    if training_args.local_rank == 0:
        print("Load tokenizer from {} over.".format(model_args.model_name_or_path))

    # Load model with memory-efficient settings
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto"  # Automatically handle device placement
    )
    
    # Resize embeddings if we added new tokens (Use in future with extended tokenizer spernado funzioni)
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        if training_args.local_rank == 0:
            print(f"Resized embeddings to {len(tokenizer)} tokens")

    # Apply LoRA to the model
    model = setup_lora_model(model, training_args)

    if training_args.local_rank == 0:
        print("LoRA model setup complete!")

    # Load and process dataset
    raw_train_datasets = load_dataset(
        'json',
        data_files=data_args.data_path,
        split="train",
        cache_dir=training_args.cache_dir
    )
    
    # Only use distributed barriers if distributed training is initialized
    if training_args.local_rank > 0 and torch.distributed.is_initialized(): 
        torch.distributed.barrier()
        
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=1000,  # Smaller batch size for processing
        num_proc=8,       # Reduced num_proc
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running Encoding",
        fn_kwargs={"tokenizer": tokenizer}
    )

    if training_args.local_rank == 0 and torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    if training_args.local_rank == 0:
        print("Training dataset samples:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 2):  # Show fewer samples
            print(f"Sample {index} of the training set:")
            print(f"Input IDs length: {len(train_dataset[index]['input_ids'])}")
            print(f"Labels length: {len(train_dataset[index]['labels'])}")
            print("Decoded text:")
            print(tokenizer.decode(train_dataset[index]['input_ids'])[:500] + "...")  # Truncate output
            print("-" * 50)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        **data_module
    )

    # Clear cache before training (added because non mi funzionava un cazz)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Train the model
    if training_args.local_rank == 0:
        print("Starting LoRA fine-tuning...")
    
    trainer.train()
    trainer.save_state()
    
    # Save LoRA adapters
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    if training_args.local_rank == 0:
        print("LoRA fine-tuning completed successfully!")
        print(f"LoRA adapters saved to: {training_args.output_dir}")

if __name__ == "__main__":
    train()