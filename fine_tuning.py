import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer



# Params
max_source_length = 128
max_target_length = 128

# Load the Parquet file into a Pandas DataFrame
df = pd.read_parquet("reactions.parquet")
# Split the data and create the train and test files 
df_train = df.iloc[:10]  # only the first 10 rows for now
df_test = df.iloc[10:15]  # only the second 5 rows for now
# Save the train and test datasets to Parquet files
df_train.to_parquet("train.parquet")
df_test.to_parquet("test.parquet")
# Load the train and test datasets int the hf format
dataset = load_dataset("parquet", data_files={'train': 'train.parquet', 'test': 'test.parquet'})

# Rename the columns according to the model's input and output
dataset = dataset.rename_column("snippet", "source")
dataset = dataset.rename_column("notes", "target")

# Quick check:
print(dataset)

# Load the model and tokenizer

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", trust_remote_code=True)

def preprocess_function(examples):
    tokenized_inputs = tokenizer(
        text=examples["source"],
        text_target=examples["target"], 
        max_length=max_source_length,         
        max_length_target=max_target_length,  
        truncation=True
    )
    return tokenized_inputs


tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True
)