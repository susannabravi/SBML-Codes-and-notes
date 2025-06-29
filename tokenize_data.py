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
    """
    Download a Hugging Face model and tokenizer to the specified directory
    """
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

def clean_data(df, columns=['snippet', 'notes']):
    """
    Remove rows where combinations of columns snippet and notes are duplicated.
    """
    initial_shape = df.shape
    print(f"Initial data shape: {initial_shape}")

    df_cleaned = df.drop_duplicates(subset=columns, keep='first')

    cleaned_shape = df_cleaned.shape
    print(f"Cleaned data shape: {cleaned_shape}")
    print(f"Removed {initial_shape[0] - cleaned_shape[0]} duplicate rows based on columns: {columns}")

    return df_cleaned

def load_or_create_train_data(parquet_file_path, train_file_path, columns=['snippet', 'notes'], force_recreate=False):
    """
    Load existing train data if available, otherwise create it from the original reactions file.
    """
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

def extend_tokenizer_vocabulary(tokenizer, new_tokens, save_path=None):
    """
    Add new tokens to the tokenizer vocabulary as special tokens to ensure they're treated as single units
    """
    print(f"Original vocabulary size: {tokenizer.vocab_size}")
    print(f"Original vocab + added tokens: {len(tokenizer.get_vocab())}")
    
    # Filter out tokens that already exist
    existing_tokens = set(tokenizer.get_vocab().keys())
    truly_new_tokens = [token for token in new_tokens if token not in existing_tokens]
    
    if not truly_new_tokens:
        print("No new tokens to add - all tokens already exist in vocabulary")
        return tokenizer, 0
    
    print(f"Adding {len(truly_new_tokens)} new tokens")
    print(f"Sample new tokens: {list(truly_new_tokens)[:10]}")
    
    # Add new tokens (this forces them to be treated as single units)
    num_added_tokens = tokenizer.add_tokens(truly_new_tokens, special_tokens=False)
    
    print(f"Successfully added {num_added_tokens} tokens")
    print(f"New vocabulary size (.vocab_size): {tokenizer.vocab_size}")
    print(f"Total vocabulary size (len(get_vocab())): {len(tokenizer.get_vocab())}")
    print(f"Difference: {len(tokenizer.get_vocab()) - tokenizer.vocab_size}")
    
    # Verify some tokens were added correctly
    sample_tokens = list(truly_new_tokens)[:5]
    print(f"\nVerifying token addition:")
    for token in sample_tokens:
        if token in tokenizer.get_vocab():
            token_id = tokenizer.get_vocab()[token]
            encoded = tokenizer.encode(token, add_special_tokens=False)
            print(f"  OK '{token}' -> ID: {token_id}, encode: {encoded}")
        else:
            print(f"  NOT OK '{token}' -> NOT FOUND in vocabulary")
    
    # Test if tokens are now single units
    print(f"\nTesting tokenization of added tokens:")
    for token in sample_tokens:
        tokens = tokenizer.tokenize(token)
        print(f"  '{token}' -> {tokens} ({'✓ single' if len(tokens) == 1 else '✗ multiple'})")
    
    # Save the extended tokenizer if path provided
    if save_path:
        print(f"Saving extended tokenizer to: {save_path}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        tokenizer.save_pretrained(save_path)
    
    return tokenizer, num_added_tokens

def test_tokenization_with_proteins(tokenizer, test_texts, title="TESTING TOKENIZATION"):
    """
    Test how the tokenizer handles protein names in context
    """
    print("\n" + "="*60)
    print(title)
    print("="*60)
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: {text}")
        
        # Tokenize
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        token_ids = tokens['input_ids'][0].tolist()
        
        # Decode individual tokens
        decoded_tokens = [tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]
        
        print(f"  Tokens ({len(token_ids)}): {decoded_tokens}")
        print(f"  Token IDs: {token_ids}")
        
        # Check specific protein names
        protein_names_in_text = re.findall(r'\b[A-Z]+[0-9]*[A-Za-z]*\b|\b[A-Za-z]+-[A-Z]+\b', text)
        for protein in protein_names_in_text:
            if len(protein) > 1:  # Skip single letters
                protein_tokens = tokenizer.tokenize(protein)
                is_single_token = len(protein_tokens) == 1
                print(f"  '{protein}' -> {protein_tokens} ({'OK, single token' if is_single_token else 'NO, multiple tokens'})")

def analyze_tokenization(df, tokenizer, text_columns=['snippet', 'notes'], sample_size=None):
    """
    Analyze tokenization of text data from a DataFrame
    """
    if sample_size:
        df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    else:
        df_sample = df
    
    for column in text_columns:
        if column not in df_sample.columns:
            print(f"Warning: Column '{column}' not found in DataFrame")
            continue
            
        print(f"\n{'-'*50}")
        print(f"ANALYZING COLUMN: {column}")
        print(f"{'-'*50}")
        
        text_data = df_sample[column].dropna()
        
        if len(text_data) == 0:
            print(f"No valid text data found in column '{column}'")
            continue
        
        token_lengths = []
        all_tokens = []
        
        print("Tokenizing texts...")
        for i, text in enumerate(text_data):
            if i % 100 == 0:
                print(f"Progress: {i+1}/{len(text_data)}")
            
            tokens = tokenizer(str(text), return_tensors="pt", truncation=False, add_special_tokens=True)
            token_ids = tokens['input_ids'][0].tolist()
            token_lengths.append(len(token_ids))
            all_tokens.extend(token_ids)
        
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
        most_common_tokens = token_counter.most_common(10)
        
        print(f"\nTOP 10 MOST FREQUENT TOKENS:")
        for token_id, count in most_common_tokens:
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            print(f"  Token ID {token_id}: '{token_text}' -> {count} times")
        
        # Show some examples
        print(f"\nSAMPLE TOKENIZATIONS:")
        sample_indices = np.random.choice(len(text_data), size=min(3, len(text_data)), replace=False)
        
        for idx in sample_indices:
            text = text_data.iloc[idx]
            tokens = tokenizer(str(text), return_tensors="pt", add_special_tokens=True)
            token_ids = tokens['input_ids'][0].tolist()
            decoded_tokens = [tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]
            
            print(f"\nOriginal text ({len(token_ids)} tokens):")
            print(f"'{str(text)[:100]}{'...' if len(str(text)) > 100 else ''}'")
            print(f"Decoded tokens: {decoded_tokens[:10]}{'...' if len(decoded_tokens) > 10 else ''}")

def main():
    # File paths
    parquet_file_path = "reactions.parquet" 
    train_file_path = "reactions_train.parquet"
    extended_tokenizer_path = "./extended_tokenizer"  
    
    # Load or create train data
    df = load_or_create_train_data(
        parquet_file_path=parquet_file_path,
        train_file_path=train_file_path,
        columns=['snippet', 'notes'],
        force_recreate=False
    )
    
    if df is None:
        print("Failed to load or create train data. Exiting.")
        return
    
    print(f"Using train data with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize tokenizer
    print("\nLoading base tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Base", trust_remote_code=True)
    
    print(f"Original tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Test original tokenizer first
    test_texts = [
        "The NOTCH3 protein interacts with JAG2 in cellular signaling.",
        "Expression of TP53 and BRCA1 was analyzed in tumor samples.",
        "TNF-α and IL-6 levels were elevated in inflammatory conditions.",
        "The α-SMA and β-catenin pathways are crucial for development."
    ]
    
    test_tokenization_with_proteins(tokenizer, test_texts, "ORIGINAL TOKENIZER TEST")
    
    print("\nExtracting protein names from data...")
    extracted_proteins = extract_protein_names_from_data(
        df, 
        text_columns=['snippet', 'notes'], 
        min_frequency=1 
    )
    
    if not extracted_proteins:
        print("No protein names extracted. Exiting.")
        return
    
    # Extend the tokenizer
    print(f"\nExtending tokenizer with {len(extracted_proteins)} protein names...")
    extended_tokenizer, num_added = extend_tokenizer_vocabulary(
        tokenizer, 
        extracted_proteins, 
        save_path=extended_tokenizer_path
    )
    
    # Test the extended tokenizer
    test_tokenization_with_proteins(extended_tokenizer, test_texts, "EXTENDED TOKENIZER TEST")
    
    # Additional test with common proteins that should be in the data
    additional_test_proteins = ["NOTCH3", "JAG2", "TP53", "BRCA1", "TNF", "IL6"]
    
    print(f"\nTesting specific proteins: {additional_test_proteins}")
    for protein in additional_test_proteins:
        # Add to tokenizer if not already there
        if protein not in extended_tokenizer.get_vocab():
            print(f"Adding missing protein: {protein}")
            extended_tokenizer.add_tokens([protein])
        
        # Test tokenization
        tokens = extended_tokenizer.tokenize(protein)
        token_ids = extended_tokenizer.encode(protein, add_special_tokens=False)
        print(f"'{protein}' -> tokens: {tokens}, ids: {token_ids}")
    
    # Analyze tokenization with extended tokenizer
    print(f"\n{'='*60}")
    print("ANALYZING TOKENIZATION WITH EXTENDED TOKENIZER")
    print(f"{'='*60}")
    
    analyze_tokenization(df, 
                        extended_tokenizer,        
                        text_columns=['snippet', 'notes'],
                        sample_size=500)  # Reduced sample size for faster processing
    
    # Final verification
    print(f"\nFinal verification - testing a few protein names:")
    test_proteins = ["NOTCH3", "TP53", "BRCA1", "JAG2"]
    for protein in test_proteins:
        if protein in extended_tokenizer.get_vocab():
            tokens = extended_tokenizer.tokenize(protein)
            print(f"  '{protein}': {tokens} ({'single token' if len(tokens) == 1 else 'multiple tokens'})")
        else:
            print(f"  '{protein}': not in vocabulary")

if __name__ == "__main__":
    main()