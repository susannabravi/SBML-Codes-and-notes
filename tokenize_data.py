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
    

if __name__ == "__main__":
    main()