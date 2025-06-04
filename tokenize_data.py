import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import flash_attn


def download_model_hf(model_path, model_name):
    """Download a Hugging Face model and tokenizer to the specified directory"""
    # Check if the directory already exists
    if not os.path.exists(model_path):
        # Create the directory
        os.makedirs(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Save the model and tokenizer to the specified directory
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


def analyze_tokenization(df, tokenizer, text_columns=['snippet', 'notes'], sample_size=None):
    """
    Analyze tokenization of text data from a DataFrame
    
    Args:
        df: pandas DataFrame containing the text data
        tokenizer: HuggingFace tokenizer
        text_columns: list of column names to analyze
        sample_size: number of samples to analyze (None for all)
    """
    if sample_size:
        df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    else:
        df_sample = df
    
    results = {}
    
    for column in text_columns:
        if column not in df_sample.columns:
            print(f"Warning: Column '{column}' not found in DataFrame")
            continue
            
        print(f"\n{'-'*50}")
        print(f"ANALYZING COLUMN: {column}")
        print(f"{'-'*50}")
        
        # Filter out null values
        text_data = df_sample[column].dropna()
        
        if len(text_data) == 0:
            print(f"No valid text data found in column '{column}'")
            continue
        
        # Tokenize all texts
        token_lengths = []
        all_tokens = []
        
        print("Tokenizing texts...")
        for i, text in enumerate(text_data):
            if i % 100 == 0:
                print(f"Progress: {i+1}/{len(text_data)}")
            
            # Tokenize the text
            tokens = tokenizer(str(text), return_tensors="pt", truncation=False)
            token_ids = tokens['input_ids'][0].tolist()
            token_lengths.append(len(token_ids))
            all_tokens.extend(token_ids)
        
        # Statistical analysis
        token_lengths = np.array(token_lengths)
        
        print(f"\nTOKENIZATION STATISTICS for {column}:")
        print(f"  Total texts: {len(text_data)}")
        print(f"  Mean token length: {token_lengths.mean():.2f}")
        print(f"  Median token length: {np.median(token_lengths):.2f}")
        print(f"  Min token length: {token_lengths.min()}")
        print(f"  Max token length: {token_lengths.max()}")
        print(f"  Std deviation: {token_lengths.std():.2f}")
        print(f"  5th percentile: {np.percentile(token_lengths, 5):.0f}")
        print(f"  25th percentile: {np.percentile(token_lengths, 25):.0f}")
        print(f"  75th percentile: {np.percentile(token_lengths, 75):.0f}")
        print(f"  95th percentile: {np.percentile(token_lengths, 95):.0f}")
        
        # Token frequency analysis
        token_counter = Counter(all_tokens)
        most_common_tokens = token_counter.most_common(20)
        
        print(f"\nTOP 20 MOST FREQUENT TOKENS:")
        for token_id, count in most_common_tokens:
            token_text = tokenizer.decode([token_id])
            print(f"  Token ID {token_id}: '{token_text}' -> {count} times")
        
        # Show some examples
        print(f"\nSAMPLE TOKENIZATIONS:")
        sample_indices = np.random.choice(len(text_data), size=min(3, len(text_data)), replace=False)
        
        for idx in sample_indices:
            text = text_data.iloc[idx]
            tokens = tokenizer(str(text), return_tensors="pt")
            token_ids = tokens['input_ids'][0].tolist()
            
            print(f"\n  Original text ({len(token_ids)} tokens):")
            print(f"    {str(text)[:200]}{'...' if len(str(text)) > 200 else ''}")
            print(f"  Token IDs: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
            print(f"  Decoded tokens: {[tokenizer.decode([tid]) for tid in token_ids[:10]]}{'...' if len(token_ids) > 10 else ''}")
        
        results[column] = {
            'token_lengths': token_lengths,
            'total_texts': len(text_data),
            'mean_length': token_lengths.mean(),
            'median_length': np.median(token_lengths),
            'max_length': token_lengths.max(),
            'min_length': token_lengths.min(),
            'std_length': token_lengths.std(),
            'most_common_tokens': most_common_tokens
        }
    
    return results

def main():

    parquet_file_path = "reactions.parquet" 
    
    try:
        df = pd.read_parquet(parquet_file_path)
        print(f"Loaded data with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Show basic info about the columns of interest
        for col in ['snippet', 'notes']:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                print(f"\nColumn '{col}': {non_null_count} non-null values out of {len(df)}")
                if non_null_count > 0:
                    avg_char_length = df[col].dropna().astype(str).str.len().mean()
                    print(f"  Average character length: {avg_char_length:.1f}")
        
    except FileNotFoundError:
        print(f"Error: Could not find parquet file at {parquet_file_path}")
        print("Please update the 'parquet_file_path' variable with the correct path to your file")
        return
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return
    
    # Initialize tokenizer and model
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", trust_remote_code=True)
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Tokenizer model max length: {tokenizer.model_max_length}")
    
    # Analyze tokenization
    results = analyze_tokenization(df, tokenizer, 
                                 text_columns=['snippet', 'notes'],
                                 sample_size=1000) 

if __name__ == "__main__":
    main()