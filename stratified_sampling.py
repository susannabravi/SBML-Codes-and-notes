import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_length_quartiles(df, column):
    lengths = df[column].apply(lambda x: len(str(x).split()))
    q1 = lengths.quantile(0.25)
    q2 = lengths.quantile(0.50)  
    q3 = lengths.quantile(0.75)
    return q1, q2, q3

def categorize_by_length(text, q1, q2, q3):
    length = len(str(text).split())
    if length <= q1:
        return "short"
    elif length <= q2:
        return "medium"
    elif length <= q3:
        return "long"
    else:
        return "very_long"

def create_stratified_categories(df):    
    # Calculate quartiles for both columns
    snippet_q1, snippet_q2, snippet_q3 = calculate_length_quartiles(df, 'snippet')
    notes_q1, notes_q2, notes_q3 = calculate_length_quartiles(df, 'notes')
    
    print("Snippet length quartiles:")
    print(f"  Q1: {snippet_q1:.0f} chars")
    print(f"  Q2: {snippet_q2:.0f} chars") 
    print(f"  Q3: {snippet_q3:.0f} chars")
    
    print("\nNotes length quartiles:")
    print(f"  Q1: {notes_q1:.0f} chars")
    print(f"  Q2: {notes_q2:.0f} chars")
    print(f"  Q3: {notes_q3:.0f} chars")
    
    # Create categories
    df_copy = df.copy()
    df_copy['snippet_category'] = df_copy['snippet'].apply(
        lambda x: categorize_by_length(x, snippet_q1, snippet_q2, snippet_q3)
    )
    df_copy['notes_category'] = df_copy['notes'].apply(
        lambda x: categorize_by_length(x, notes_q1, notes_q2, notes_q3)
    )
    
    # Combine categories
    df_copy['combined_category'] = df_copy['snippet_category'] + '_' + df_copy['notes_category']
    
    return df_copy

def visualize_distribution(df: pd.DataFrame, save_path: str = None):

    plt.style.use('ggplot')
    
    plt.figure(figsize=(12, 8))

    crosstab = pd.crosstab(df['snippet_category'], df['notes_category'])
    print(crosstab)
    
    # Use ggplot style default colors
    all_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ggplot_colors = all_colors[1:5]
    crosstab.plot(kind='bar', stacked=True, color=ggplot_colors)
    
    plt.title('Distribution of Samples Across Length Categories')
    plt.xlabel('Snippet Length Category')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    
    plt.legend(title='Notes Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    
def print_stats(df):
    # Print distribution summary
    print("\nDistribution of combined categories:")
    category_summary = df['combined_category'].value_counts().sort_index()
    print(category_summary)
    
    # Print length statistics
    print("\nLength Statistics:")
    print(f"Snippet lengths - Min: {df['snippet'].str.len().min()}, Max: {df['snippet'].str.len().max()}, Mean: {df['snippet'].str.len().mean():.1f}")
    print(f"Notes lengths - Min: {df['notes'].str.len().min()}, Max: {df['notes'].str.len().max()}, Mean: {df['notes'].str.len().mean():.1f}")
    
    # Print category balance
    print(f"\nCategory Balance:")
    print(f"Most common category: {category_summary.index[-1]} ({category_summary.iloc[-1]} samples)")
    print(f"Least common category: {category_summary.index[0]} ({category_summary.iloc[0]} samples)")
    print(f"Balance ratio: {category_summary.iloc[-1] / category_summary.iloc[0]:.2f}:1")

def stratified_split(df, train_size = 0.7, val_size = 0.15, test_size = 0.15, random_state = 42):
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(val_size + test_size), 
        stratify=df['combined_category'],
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        stratify=temp_df['combined_category'],
        random_state=random_state
    )
    
    return train_df, val_df, test_df

def prepare_for_finetuning(df: pd.DataFrame, instruction_template: str = None) -> pd.DataFrame:
    """
    Prepare the data in the format expected by the fine-tuning script.
    """
    
    if instruction_template is None:
        instruction_template = "Translate the following SBML code to English:\n\n{snippet}"
    
    # Create the fine-tuning format
    prepared_df = pd.DataFrame()
    prepared_df['instruction'] = df['snippet'].apply(
        lambda x: instruction_template.format(snippet=x)
    )
    prepared_df['output'] = df['notes']
    
    return prepared_df

def save_datasets(train_df: pd.DataFrame, 
                 val_df: pd.DataFrame, 
                 test_df: pd.DataFrame,
                 output_dir: str = "./"):
    """Save datasets in JSON format for fine-tuning."""

    os.makedirs(output_dir, exist_ok=True)
    # Prepare datasets for fine-tuning
    train_prepared = prepare_for_finetuning(train_df)
    val_prepared = prepare_for_finetuning(val_df)
    test_prepared = prepare_for_finetuning(test_df)
    
    # Save as JSON files
    train_prepared.to_json(f"{output_dir}/train_data.json", orient='records', lines=True)
    val_prepared.to_json(f"{output_dir}/val_data.json", orient='records', lines=True)
    test_prepared.to_json(f"{output_dir}/test_data.json", orient='records', lines=True)
    
    # Also save the original splits with category information
    train_df.to_json(f"{output_dir}/train_with_categories.json", orient='records', lines=True)
    val_df.to_json(f"{output_dir}/val_with_categories.json", orient='records', lines=True)
    test_df.to_json(f"{output_dir}/test_with_categories.json", orient='records', lines=True)
    
    print(f"\nDatasets saved to {output_dir}:")
    print(f"  Training set: {len(train_prepared)} samples")
    print(f"  Validation set: {len(val_prepared)} samples")
    print(f"  Test set: {len(test_prepared)} samples")

def main():
    """Main function to run the stratified sampling pipeline."""
    
    df = pd.read_parquet("./dataset_with_token_counts.parquet")
    
    # Example of how to use with your data:
    print("Loading data...")
    
    print(f"Loaded {len(df)} samples")
    
    # Create stratified categories
    print("\nCreating stratified categories...")
    df_categorized = create_stratified_categories(df)
    
    # Visualize distribution
    print("\nVisualizing distribution...")
    visualize_distribution(df_categorized, save_path="category_distribution.png")
    print_stats(df_categorized)

    # Perform stratified split
    print("\nPerforming stratified split")
    train_df, val_df, test_df = stratified_split(
        df_categorized, 
        train_size=0.7, 
        val_size=0.15, 
        test_size=0.15,
        random_state=42
    )
    
    # Verify stratification worked
    print("\nVerifying stratification")
    print("Training set distribution:")
    print(train_df['combined_category'].value_counts().sort_index())
    print("\nValidation set distribution:")
    print(val_df['combined_category'].value_counts().sort_index())
    print("\nTest set distribution:")
    print(test_df['combined_category'].value_counts().sort_index())
    
    # Save datasets
    print("\nSaving datasets...")
    save_datasets(train_df, val_df, test_df, output_dir="./finetune_data")
    
    print("\nStratified sampling completed successfully!")

if __name__ == "__main__":
    main()