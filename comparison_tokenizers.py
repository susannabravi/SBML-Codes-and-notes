import pandas as pd
import nltk
from nltk.tokenize import word_tokenize  #https://www.geeksforgeeks.org/nlp/tokenize-text-using-nltk-python/
from transformers import AutoTokenizer

nltk.download('punkt_tab')

# Load dataset
df = pd.read_parquet("./reactions_train.parquet")

# Load the DeepSeek tokenizer 
deepseek_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Base", trust_remote_code=True)
    
def tokenization_nltk(text):
    if pd.isna(text):
        return [], 0 
    tokens = word_tokenize(str(text))
    return tokens, len(tokens)

def tokenization_deep(text):
    if pd.isna(text):
        return [], 0
    
    tokens = deepseek_tokenizer.tokenize(str(text))
    return tokens, len(tokens)

# Function to process text columns
def add_word_counts(dataframe, col, tokenizer, name):
    """Add word count columns for specified text columns"""
    
    # Apply to each specified column
    if col in dataframe.columns:
        print(f"Processing {len(dataframe)} rows with {name} ")
        
        # Get tokens and counts
        tokenization_results = dataframe[col].apply(tokenizer)
        
        # Split into separate columns
        dataframe[f'{col}_tokens_{name}'] = tokenization_results.apply(lambda x: x[0])
        dataframe[f'{col}_token_count_{name}'] = tokenization_results.apply(lambda x: x[1])
        
        print(f"Finished processing with {name} tokenizer")
    else:
        print(f"Warning: Column '{col}' not found in dataframe")
    
    return dataframe

text_columns = 'notes' 

# Apply word counting
df = add_word_counts(df, text_columns, tokenization_nltk, "nltk")
df = add_word_counts(df, text_columns, tokenization_deep, "deep")

# Save results as parquet file
df.to_parquet('dataset_with_token_counts.parquet', index=False)

print("Finishhh.")