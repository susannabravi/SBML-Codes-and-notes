import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import flash_attn
import re


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
    
def extend_tokenizer_vocabulary(tokenizer, new_tokens, save_path=None):
    """
    Add new tokens to the tokenizer vocabulary
    """

    print(f"Original vocabulary size: {tokenizer.vocab_size}")
    
    # Filter out tokens that already exist
    existing_tokens = set(tokenizer.get_vocab().keys())
    truly_new_tokens = [token for token in new_tokens if token not in existing_tokens]
    
    if not truly_new_tokens:
        print("No new tokens to add - all tokens already exist in vocabulary")
        return tokenizer, 0
    
    print(f"Adding {len(truly_new_tokens)} new tokens: {truly_new_tokens}")
    
    # Add new tokens to the tokenizer
    num_added_tokens = tokenizer.add_tokens(truly_new_tokens)
    
    print(f"Successfully added {num_added_tokens} tokens")
    print(f"New vocabulary size: {tokenizer.vocab_size}")
    
    # Save the extended tokenizer if path provided
    if save_path:
        print(f"Saving extended tokenizer to: {save_path}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        tokenizer.save_pretrained(save_path)
    
    return tokenizer, num_added_tokens    

def extract_protein_names_from_data(df, text_column=['notes'], pattern=None):
    """
    Extract protein names from the dataset using regex patterns
    """
    if pattern is None:
        # Common protein name patterns:
        # - All caps with numbers: NOTCH3, JAG2, TP53
        # - Mixed case with numbers: Notch3, p53
        # - Greek letters: α-SMA, β-catenin
        pattern = r'\b[A-Za-z]*[A-Z]+[A-Za-z]*\d+[A-Za-z]*\b|\b[A-Z]{2,}[0-9]*\b|\b[A-Za-z]*[αβγδεζηθικλμνξοπρστυφχψω]-?[A-Za-z]+\b'
    
    protein_names = set()
     
    print(f"Extracting protein names from column: {text_column}")
    text_data = df[text_column].dropna()
    
    for text in text_data:
        matches = re.findall(pattern, str(text))
        # Filter matches to reasonable protein name lengths and formats
        filtered_matches = [
            match for match in matches 
            if 2 <= len(match) <= 20 and not match.isdigit()
        ]
        protein_names.update(filtered_matches)

    return protein_names





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

def test_tokenization_with_proteins(tokenizer, test_texts):
    """
    Test how the tokenizer handles protein names in context
    """
    print("\n" + "="*60)
    print("TESTING TOKENIZATION WITH PROTEIN NAMES")
    print("="*60)
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: {text}")
        
        # Tokenize
        tokens = tokenizer(text, return_tensors="pt")
        token_ids = tokens['input_ids'][0].tolist()
        
        # Decode individual tokens
        decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]
        
        print(f"  Tokens ({len(token_ids)}): {decoded_tokens}")
        print(f"  Token IDs: {token_ids}")
        
        # Check if protein names are single tokens
        protein_names_in_text = re.findall(r'\b[A-Z]+\d*\b', text)
        for protein in protein_names_in_text:
            protein_tokens = tokenizer.tokenize(protein)
            is_single_token = len(protein_tokens) == 1
            print(f"  '{protein}' -> {protein_tokens} ({'✓ single token' if is_single_token else '✗ multiple tokens'})")

def main():
    # File paths
    parquet_file_path = "reactions.parquet" 
    train_file_path = "reactions_train.parquet"
    extended_tokenizer_path = "./extended_tokenizer"  # Path to save extended tokenizer
    
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
    print(f"Original tokenizer model max length: {tokenizer.model_max_length}")
    
    # Option 1: Add custom curated protein list
    print("\n" + "="*60)
    print("EXTENDING TOKENIZER WITH CUSTOM PROTEIN NAMES")
    print("="*60)
    
    # Option 2: Extract protein names from your data (uncomment to use)
    print("\nExtracting protein names from data...")
    extracted_proteins = extract_protein_names_from_data(df, ['snippet', 'notes'])
    print(f"Extracted {len(extracted_proteins)} unique protein-like terms from data")
    print(f"Sample extracted: {list(extracted_proteins)[:10] if extracted_proteins else 'None'}")
    
    # Extend the tokenizer
    extended_tokenizer, num_added = extend_tokenizer_vocabulary(
        tokenizer, 
        extracted_proteins, 
        save_path=extended_tokenizer_path
    )
    
    # Test the extended tokenizer
    test_texts = [
        "The NOTCH3 protein interacts with JAG2 in cellular signaling.",
        "Expression of TP53 and BRCA1 was analyzed in tumor samples.",
        "TNF-α and IL-6 levels were elevated in inflammatory conditions.",
        "The α-SMA and β-catenin pathways are crucial for development."
    ]
    
    test_tokenization_with_proteins(extended_tokenizer, test_texts)
    
    # Show basic info about the columns of interest
    for col in ['snippet', 'notes']:
        if col in df.columns:
            non_null_count = df[col].notna().sum()
            print(f"\nColumn '{col}': {non_null_count} non-null values out of {len(df)}")
    
    # Analyze tokenization with extended tokenizer
    print(f"\n{'='*60}")
    print("ANALYZING TOKENIZATION WITH EXTENDED TOKENIZER")
    print(f"{'='*60}")
    
    results = analyze_tokenization(df, 
                                   extended_tokenizer,        
                                   text_columns=['snippet', 'notes'],
                                   sample_size=1000) 
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Original vocabulary size: {tokenizer.vocab_size}")
    print(f"Extended vocabulary size: {extended_tokenizer.vocab_size}")
    print(f"Added {num_added} new tokens")
    print(f"Extended tokenizer saved to: {extended_tokenizer_path}")

if __name__ == "__main__":
    main()