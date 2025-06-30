import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re


def download_model_hf(model_path, model_name):
    # Download a Hugging Face model and tokenizer to the specified directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

def clean_data(df, columns=['snippet', 'notes']):
    #Remove rows where combinations of columns snippet and notes are duplicated.
    initial_shape = df.shape
    print(f"Initial data shape: {initial_shape}")

    df_cleaned = df.drop_duplicates(subset=columns, keep='first')

    cleaned_shape = df_cleaned.shape
    print(f"Cleaned data shape: {cleaned_shape}")
    print(f"Removed {initial_shape[0] - cleaned_shape[0]} duplicate rows based on columns: {columns}")

    return df_cleaned

def load_or_create_train_data(parquet_file_path, train_file_path, columns=['snippet', 'notes'], force_recreate=False):
    #Load existing train data if available, otherwise create it from the original reactions file.
    if os.path.exists(train_file_path) and not force_recreate:
        print(f"Loading existing train data from: {train_file_path}")
        try:
            df_train = pd.read_parquet(train_file_path)
            print(f"Loaded train data with shape: {df_train.shape}")
            return df_train
        except Exception as e:
            print(f"Error loading existing train file: {e}")
            print("Will create new train file...")
    
    print(f"Creating new train data from: {parquet_file_path}")
    try:
        df = pd.read_parquet(parquet_file_path)
        print(f"Loaded original data with shape: {df.shape}")
        
        df_cleaned = clean_data(df, columns=columns)
        
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

def load_protein_list(protein_file_path):
    #Load protein names from text file
    with open(protein_file_path, 'r', encoding='utf-8') as f:
        proteins = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(proteins)} proteins from {protein_file_path}")
    return proteins

def extend_tokenizer_vocabulary(tokenizer, new_tokens, save_path=None):
    # Add new tokens to the tokenizer vocabulary as special tokens to ensure they're treated as single units

    print(f"Original vocabulary size: {tokenizer.vocab_size}")
    
    # Filter out tokens that already exist
    existing_tokens = set(tokenizer.get_vocab().keys())
    truly_new_tokens = [token for token in new_tokens if token not in existing_tokens]
    
    if not truly_new_tokens:
        print("No new tokens to add - all tokens already exist in vocabulary")
        return tokenizer, 0
    
    print(f"Adding {len(truly_new_tokens)} new tokens")
    num_added_tokens = tokenizer.add_tokens(truly_new_tokens, special_tokens=False)
    
    print(f"Successfully added {num_added_tokens} tokens")
    print(f"Total vocabulary size (len(get_vocab())): {len(tokenizer.get_vocab())}")
    print(f"Difference: {len(tokenizer.get_vocab()) - tokenizer.vocab_size}")
    
    # Save the extended tokenizer if path provided
    if save_path:
        print(f"Saving extended tokenizer to: {save_path}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        tokenizer.save_pretrained(save_path)
    
    return tokenizer, num_added_tokens

def main():
    # File paths
    parquet_file_path = "reactions.parquet" 
    train_file_path = "reactions_train.parquet"
    protein_file_path = "reactome_proteins_clean.txt"
    extended_tokenizer_path = "./extended_tokenizer"
    
    base_model_name = "deepseek-ai/DeepSeek-Coder-V2-Base"  
    
    print("=" * 50)
    print("CREATING EXTENDED TOKENIZER WITH PROTEIN VOCABULARY")
    print("=" * 50)
    
    # Load or create train data
    df = load_or_create_train_data(
        parquet_file_path=parquet_file_path,
        train_file_path=train_file_path,
        columns=['snippet', 'notes'],
        force_recreate=False
    )

    # Load protein list from text file
    print(f"\nLoading protein list from: {protein_file_path}")
    protein_list = load_protein_list(protein_file_path)
    
    if not protein_list:
        print("No proteins loaded. Exiting...")
        return
    
    # Load base tokenizer
    print(f"\nLoading base tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Extend tokenizer with protein vocabulary
    print(f"\nExtending tokenizer with {len(protein_list)} proteins...")
    extended_tokenizer, num_added = extend_tokenizer_vocabulary(
        tokenizer=tokenizer,
        new_tokens=protein_list,
        save_path=extended_tokenizer_path
    )

if __name__ == "__main__":
    main()