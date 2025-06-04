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
    """
    Download a Hugging Face model and tokenizer to the specified directory
    """
    # Check if the directory already exists
    if not os.path.exists(model_path):
        # Create the directory
        os.makedirs(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Save the model and tokenizer to the specified directory
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

def clean_data(df, columns=['snippet', 'notes']):
    """
    Remove rows where combinations of columns snippet and notes are duplicated.
    
    Args:
        df: pandas DataFrame containing the text data
        columns: list of column names to use for duplicate detection 

    Returns:
        Cleaned DataFrame with duplicates removed.
    """
    initial_shape = df.shape
    print(f"Initial data shape: {initial_shape}")

    # Drop duplicate rows based on specified columns
    df_cleaned = df.drop_duplicates(subset=columns, keep='first')

    cleaned_shape = df_cleaned.shape
    print(f"Cleaned data shape: {cleaned_shape}")
    print(f"Removed {initial_shape[0] - cleaned_shape[0]} duplicate rows based on columns: {columns}")

    return df_cleaned

def load_or_create_train_data(parquet_file_path, train_file_path, columns=['snippet', 'notes'], force_recreate=False):
    """
    Load existing train data if available, otherwise create it from the original parquet file.
    
    Args:
        parquet_file_path: path to the original parquet file
        train_file_path: path to save/load the cleaned train data
        columns: columns to use for duplicate detection during cleaning
        force_recreate: if True, recreate the train file even if it exists
    
    Returns:
        Cleaned DataFrame ready for training
    """
    # Check if train file already exists and we don't want to force recreate
    if os.path.exists(train_file_path) and not force_recreate:
        print(f"Loading existing train data from: {train_file_path}")
        try:
            df_train = pd.read_parquet(train_file_path)
            print(f"Loaded train data with shape: {df_train.shape}")
            return df_train
        except Exception as e:
            print(f"Error loading existing train file: {e}")
            print("Will create new train file...")
    
    # Load original data and clean it
    print(f"Creating new train data from: {parquet_file_path}")
    try:
        df = pd.read_parquet(parquet_file_path)
        print(f"Loaded original data with shape: {df.shape}")
        
        # Clean the data
        df_cleaned = clean_data(df, columns=columns)
        
        # Save the cleaned data as train file
        print(f"Saving cleaned train data to: {train_file_path}")
        df_cleaned.to_parquet(train_file_path, index=False)
        
        print(f"Train file created successfully!")
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: Could not find original parquet file at {parquet_file_path}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None


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
    # File paths
    parquet_file_path = "reactions.parquet" 
    train_file_path = "reactions_train.parquet"  # Cleaned data for training
    
    # Load or create train data (set force_recreate=True to always recreate)
    df = load_or_create_train_data(
        parquet_file_path=parquet_file_path,
        train_file_path=train_file_path,
        columns=['snippet', 'notes'],
        force_recreate=False  # Change to True if you want to force recreation
    )
    
    if df is None:
        print("Failed to load or create train data. Exiting.")
        return
    
    print(f"Using train data with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Show basic info about the columns of interest
    for col in ['snippet', 'notes']:
        if col in df.columns:
            non_null_count = df[col].notna().sum()
            print(f"\nColumn '{col}': {non_null_count} non-null values out of {len(df)}")
    
    # Initialize tokenizer and model
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Base", trust_remote_code=True)
    
    # Prints the total number of tokens (words, subwords, symbols) in the tokenizer's vocabulary
    # --> How many unique tokens the tokenizer can handle
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    # Prints the maximum number of tokens the model can process in a single input sequence 
    # --> How long the input sequences can be for the model 
    print(f"Tokenizer model max length: {tokenizer.model_max_length}") 
    
    # Analyze tokenization
    results = analyze_tokenization(df, 
                                   tokenizer,        
                                   text_columns=['snippet', 'notes'],
                                   sample_size=None) 

if __name__ == "__main__":
    main()