import copy
import random
import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from datetime import datetime

import torch
import torch.distributed
import transformers
from transformers import Trainer, BitsAndBytesConfig, TrainerCallback
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


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
    use_dora: bool = field(default=True, metadata={"help": "Use DoRA instead of LoRA"})
    use_quantization: bool = field(default=False, metadata={"help": "Use 4-bit quantization (QDoRA)"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data (JSON file)."})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "Path to eval data (optional)"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    
    # Training parameters optimized for DoRA
    learning_rate: float = field(default=2e-4)  # Slightly lower for DoRA
    warmup_ratio: float = field(default=0.03)
    weight_decay: float = field(default=0.01)  # Small weight decay for DoRA
    
    # Memory optimization
    gradient_checkpointing: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=8)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    dataloader_pin_memory: bool = field(default=False)
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    
    # DoRA/LoRA configuration
    lora_r: int = field(default=32, metadata={"help": "LoRA/DoRA rank"})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA/DoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA/DoRA dropout"})
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of target modules. Default: attention + MLP layers"}
    )
    
    # Saving and evaluation
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    evaluation_strategy: str = field(default="no")  # Will be set to "steps" if eval data provided
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=False)  # Will be set to True if eval data provided
    metric_for_best_model: str = field(default="loss")
    greater_is_better: bool = field(default=False)  # For loss, smaller is better
    
    # Logging configuration - UPDATED for terminal output
    report_to: str = field(default="none")  # Disable wandb/tensorboard
    logging_dir: Optional[str] = field(default="./logs")
    logging_steps: int = field(default=10)  # Log every 10 steps
    logging_first_step: bool = field(default=True)  # Log the first step
    logging_strategy: str = field(default="steps")  # Always log based on steps
    log_level: str = field(default="info")  # Set logging level to info
    disable_tqdm: bool = field(default=False)  # Keep progress bar enabled
    
    # Disable hub
    push_to_hub: bool = field(default=False)
    hub_model_id: Optional[str] = field(default=None)
    
    # Custom logging to file
    save_logs_to_file: bool = field(default=True, metadata={"help": "Save training logs to JSON file"})


class CustomLoggingCallback(TrainerCallback):
    """Custom callback to log training metrics to console and optionally to file."""
    
    def __init__(self, save_to_file=True, output_dir="./"):
        self.save_to_file = save_to_file
        self.output_dir = output_dir
        self.log_history = []
        
        if self.save_to_file:
            self.log_file = os.path.join(output_dir, "training_log.json")
            print(f"Training logs will be saved to: {self.log_file}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging happens."""
        if logs is None:
            return
        
        # Add timestamp
        logs["timestamp"] = datetime.now().isoformat()
        logs["global_step"] = state.global_step
        logs["epoch"] = state.epoch
        
        # Format the output for terminal
        output_parts = [f"\n[Step {state.global_step}]"]
        
        # Add epoch info if available
        if state.epoch is not None:
            output_parts.append(f"Epoch: {state.epoch:.2f}")
        
        # Add loss info
        if "loss" in logs:
            output_parts.append(f"Training Loss: {logs['loss']:.4f}")
        
        # Add eval metrics if available
        if "eval_loss" in logs:
            output_parts.append(f"Validation Loss: {logs['eval_loss']:.4f}")
        
        # Add learning rate if available
        if "learning_rate" in logs:
            output_parts.append(f"Learning Rate: {logs['learning_rate']:.2e}")
        
        # Add gradient norm if available  
        if "grad_norm" in logs:
            output_parts.append(f"Gradient Norm: {logs['grad_norm']:.4f}")
        
        # Print to console with clear formatting
        if len(output_parts) > 1:  # Only print if we have actual metrics
            print(" | ".join(output_parts))
            print("-" * 80)  # Separator line
        
        # Save to file if enabled
        if self.save_to_file:
            self.log_history.append(logs)
            with open(self.log_file, 'w') as f:
                json.dump(self.log_history, f, indent=2, default=str)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        print("\n" + "="*80)
        print("TRAINING COMPLETED!")
        print("="*80)
        print(f"Total steps: {state.global_step}")
        print(f"Total epochs: {state.epoch:.2f}")
        
        if self.save_to_file and self.log_history:
            # Save final summary
            summary = {
                "training_completed": datetime.now().isoformat(),
                "total_steps": state.global_step,
                "total_epochs": state.epoch,
                "final_train_loss": self.log_history[-1].get("loss", "N/A"),
                "final_eval_loss": self.log_history[-1].get("eval_loss", "N/A")
            }
            
            summary_file = os.path.join(self.output_dir, "training_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nTraining summary saved to: {summary_file}")
            print(f"Full training log saved to: {self.log_file}")
    
    def on_evaluation(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation phase."""
        if metrics:
            print("\n" + "="*80)
            print("EVALUATION RESULTS:")
            for key, value in metrics.items():
                if key.startswith("eval_"):
                    metric_name = key.replace("eval_", "").replace("_", " ").title()
                    print(f"  {metric_name}: {value:.4f}")
            print("="*80 + "\n")


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
    """Tokenize training examples."""
    sources = [
        build_instruction_prompt(instruction)
        for instruction in examples['instruction']
    ]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['output']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


def setup_tokenizer(model_path: str, model_max_length: int):
    """Setup tokenizer for DeepSeek base model with proper special tokens."""
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


def setup_dora_model(model, model_args, training_args):
    """Setup DoRA/LoRA configuration and apply to model."""
    
    # Parse target modules
    if training_args.lora_target_modules:
        target_modules = training_args.lora_target_modules.split(",")
    else:
        # Default modules for DeepSeek Coder V2
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
            "gate_proj", "up_proj", "down_proj"      # MLP layers  
        ]
    
    # LoRA/DoRA configuration
    from peft import LoraConfig as PeftLoraConfig
    
    # Check if this version of PEFT supports DoRA
    import inspect
    lora_init_params = inspect.signature(PeftLoraConfig.__init__).parameters
    
    # For non-quantized models, we'll only save lm_head to avoid gradient issues
    # but not embed_tokens (which is the really large part)
    modules_to_save_list = ["lm_head"] if not model_args.use_quantization else ["embed_tokens", "lm_head"]
    
    if 'use_dora' in lora_init_params:
        # PEFT supports DoRA directly in constructor
        lora_config = PeftLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            use_dora=model_args.use_dora,
            modules_to_save=modules_to_save_list
        )
        print(f"Using {'DoRA' if model_args.use_dora else 'LoRA'} configuration")
    else:
        # Fallback: create config without use_dora
        lora_config = PeftLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            modules_to_save=modules_to_save_list
        )
        # Try to set use_dora as attribute if supported
        if hasattr(lora_config, 'use_dora'):
            lora_config.use_dora = model_args.use_dora
        else:
            print("Warning: This version of PEFT doesn't support DoRA. Using standard LoRA.")
            print("Please upgrade PEFT: pip install --upgrade peft")
    
    # Apply LoRA/DoRA to the model
    model = get_peft_model(model, lora_config)
    
    # Print configuration
    print(f"\n{'DoRA' if model_args.use_dora else 'LoRA'} Configuration:")
    print(f"  Rank: {training_args.lora_r}")
    print(f"  Alpha: {training_args.lora_alpha}")
    print(f"  Dropout: {training_args.lora_dropout}")
    print(f"  Target modules: {target_modules}")
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing for memory efficiency
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model


def load_model(model_args, training_args):
    """Load model with optional quantization for QDoRA."""
    
    if model_args.use_quantization:
        print("Loading model with 4-bit quantization (QDoRA)...")
        
        # Quantization config for QDoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        
    else:
        print("Loading model without quantization...")
        
        # Determine compute dtype
        compute_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)
        
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    
    return model


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"  # Disable wandb completely
    
    # Set up logging to console
    import logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    
    print('='*100)
    print(f"Fine-tuning DeepSeek Coder V2 with {'DoRA' if model_args.use_dora else 'LoRA'}")
    if model_args.use_quantization:
        print("Using 4-bit quantization (QDoRA)")
    print(f"Model: {model_args.model_name_or_path}")
    print('='*100)
    
    # Setup tokenizer
    tokenizer = setup_tokenizer(model_args.model_name_or_path, training_args.model_max_length)
    
    print(f"\nTokenizer setup complete:")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    # Load model with optional quantization
    model = load_model(model_args, training_args)
    
    # Resize embeddings if we added new tokens
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized embeddings to {len(tokenizer)} tokens")
    
    # Apply DoRA/LoRA to the model
    model = setup_dora_model(model, model_args, training_args)
    
    # Load and process training dataset
    print(f"\nLoading dataset from: {data_args.data_path}")
    raw_train_datasets = load_dataset(
        'json',
        data_files=data_args.data_path,
        split="train",
        cache_dir=training_args.cache_dir
    )
    
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=8,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Tokenizing training data",
        fn_kwargs={"tokenizer": tokenizer}
    )
    
    # Load eval dataset if provided
    eval_dataset = None
    if data_args.eval_data_path:
        print(f"\nLoading eval dataset from: {data_args.eval_data_path}")
        raw_eval_datasets = load_dataset(
            'json',
            data_files=data_args.eval_data_path,
            split="train",
            cache_dir=training_args.cache_dir
        )
        
        eval_dataset = raw_eval_datasets.map(
            train_tokenize_function,
            batched=True,
            batch_size=1000,
            num_proc=8,
            remove_columns=raw_eval_datasets.column_names,
            load_from_cache_file=True,
            desc="Tokenizing eval data",
            fn_kwargs={"tokenizer": tokenizer}
        )
        print(f"Eval dataset samples: {len(eval_dataset)}")
        
        # Enable evaluation if eval dataset is provided
        training_args.evaluation_strategy = "steps"
        training_args.load_best_model_at_end = True
        print("Evaluation enabled with eval dataset")
    
    print(f"\nTraining dataset samples: {len(train_dataset)}")
    
    # Calculate and display training statistics
    total_steps = len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs
    print(f"\nTraining Configuration:")
    print(f"  Total training samples: {len(train_dataset)}")
    print(f"  Batch size per device: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Number of epochs: {training_args.num_train_epochs}")
    print(f"  Total optimization steps: {total_steps}")
    print(f"  Logging every {training_args.logging_steps} steps")
    if eval_dataset:
        print(f"  Evaluating every {training_args.eval_steps} steps")
    print(f"  Saving checkpoints every {training_args.save_steps} steps")
    
    # Show sample data
    print("\nSample training data:")
    for index in random.sample(range(min(len(train_dataset), 10)), min(2, len(train_dataset))):
        print(f"\n  Sample {index}:")
        print(f"    Input length: {len(train_dataset[index]['input_ids'])}")
        print(f"    Sample text (first 300 chars):")
        decoded = tokenizer.decode(train_dataset[index]['input_ids'])
        print(f"    {decoded[:300]}...")
    
    # Setup data collator
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # Create custom logging callback
    custom_callback = CustomLoggingCallback(
        save_to_file=training_args.save_logs_to_file,
        output_dir=training_args.output_dir
    )
    
    # Initialize trainer with custom callback
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[custom_callback],  # Add custom callback
    )
    
    # Clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\nGPU memory cleared. Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, Memory: {props.total_memory / 1024**3:.1f} GB")
    
    # Train the model
    print(f"\n{'='*100}")
    print(f"Starting {'DoRA' if model_args.use_dora else 'LoRA'} fine-tuning...")
    print(f"{'='*100}\n")
    
    train_result = trainer.train()
    
    # Print training results
    print("\n" + "="*100)
    print("Training Results:")
    print(f"  Total training time: {train_result.metrics['train_runtime']:.2f} seconds")
    print(f"  Training samples per second: {train_result.metrics['train_samples_per_second']:.2f}")
    print(f"  Training steps per second: {train_result.metrics['train_steps_per_second']:.2f}")
    print(f"  Total FLOPs: {train_result.metrics.get('total_flos', 'N/A')}")
    print(f"  Final training loss: {train_result.metrics['train_loss']:.4f}")
    print("="*100)
    
    # Save the final model
    trainer.save_model()
    trainer.save_state()
    
    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Run final evaluation if eval dataset exists
    if eval_dataset:
        print("\nRunning final evaluation...")
        eval_results = trainer.evaluate()
        
        print("\nFinal Evaluation Results:")
        for key, value in eval_results.items():
            if key.startswith("eval_"):
                metric_name = key.replace("eval_", "").replace("_", " ").title()
                print(f"  {metric_name}: {value:.4f}")
        
        # Save evaluation results
        eval_results_file = os.path.join(training_args.output_dir, "final_eval_results.json")
        with open(eval_results_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nEvaluation results saved to: {eval_results_file}")
    
    print("\n" + "="*100)
    print(f"{'DoRA' if model_args.use_dora else 'LoRA'} fine-tuning completed successfully!")
    print(f"Model saved to: {training_args.output_dir}")
    if model_args.use_quantization:
        print("Note: Model was trained with 4-bit quantization (QDoRA)")
    print("="*100)


if __name__ == "__main__":
    train()