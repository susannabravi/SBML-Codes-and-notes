import pandas as pd
import pysbd
import re
from collections import Counter

# First, let's debug what's happening with your current data
def debug_sentence_issues(df_path="./sentences.parquet"):
    """Debug why we're getting periods as sentences"""
    
    df_sentences = pd.read_parquet(df_path)
    
    print("=== DEBUGGING SENTENCE SEGMENTATION ISSUES ===\n")
    
    # Find all problematic sentences
    problematic = df_sentences[df_sentences['sentence'].str.len() <= 3]['sentence'].value_counts()
    print("Problematic sentences (3 characters or less):")
    for sentence, count in problematic.head(10).items():
        print(f"  '{sentence}' appears {count} times")
    
    print(f"\nTotal problematic sentences: {len(problematic)}")
    print(f"Total occurrences: {problematic.sum()}")
    
    # Let's look at some original notes that produce these periods
    # Let's look at some original notes that produce these periods
    period_rows = df_sentences[df_sentences['sentence'] == '.']
    if len(period_rows) > 0:
        print(f"\n=== NOTES THAT PRODUCE STANDALONE PERIODS ===")
        print(f"Found {len(period_rows)} rows with standalone periods")
        
        # Get sample notes that contain periods
        sample_notes = period_rows['notes'].unique()[:5]
        for i, note in enumerate(sample_notes, 1):
            print(f"\nSample note {i}:")
            print(f"Notes: '{note}'")
            
            # Get corresponding original_notes for this note
            matching_rows = period_rows[period_rows['notes'] == note]
            original_notes = matching_rows['original_notes'].iloc[0]  # Get first matching original_notes
            print(f"Original Notes: '{original_notes}'")
            
            print(f"Notes Length: {len(note)}")
            print(f"Original Notes Length: {len(original_notes)}")
            
            # Test sentence segmentation on this note
            segmenter = pysbd.Segmenter(language="en", clean=True)
            sentences = segmenter.segment(note)
            print(f"Segmented into {len(sentences)} sentences:")
            for j, sent in enumerate(sentences):
                print(f"  {j+1}: '{sent}' (len: {len(sent)})")
            
            print("-" * 50)
    
    return df_sentences

# Run the debug function
debug_sentence_issues()

print("\n" + "="*50)
print("IMPROVED SENTENCE PROCESSING")
print("="*50)

def robust_sentence_filter(text):
    """More robust sentence filtering"""
    if not text or not isinstance(text, str):
        return ""
    
    text = text.strip()
    
    # Skip empty or whitespace-only
    if not text:
        return ""
    
    # Skip single characters or just punctuation
    if len(text) <= 2:
        return ""
    
    # Skip if it's just punctuation and whitespace
    if re.match(r'^[\s\W]+$', text):
        return ""
    
    # Skip if it has no letters (just numbers and punctuation)
    if not re.search(r'[a-zA-Z]', text):
        return ""
    
    # Skip very short "sentences" that are likely artifacts
    if len(text.split()) < 2 and len(text) < 10:
        return ""
    
    return text

def improved_sentence_processing(df_path="./reactions.parquet"):
    """Improved sentence processing pipeline"""
    
    # Load and clean original data
    df = pd.read_parquet(df_path)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != 'file_id']).reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Extract and deduplicate notes
    df2 = df[['notes']].drop_duplicates().reset_index(drop=True)
    
    print(f"Processing {len(df2)} unique notes...")
    
    # Initialize sentence segmenter
    segmenter = pysbd.Segmenter(language="en", clean=True)
    
    def process_note(text):
        """Process a single note into clean sentences"""
        if not text or not isinstance(text, str):
            return []
        
        # Segment the text
        sentences = segmenter.segment(text)
        
        # Filter each sentence
        clean_sentences = []
        for sentence in sentences:
            clean_sentence = robust_sentence_filter(sentence)
            if clean_sentence:  # Only add non-empty sentences
                clean_sentences.append(clean_sentence)
        
        return clean_sentences
    
    # Process all notes
    df2['sentences'] = df2['notes'].apply(process_note)
    df2['num_sentences'] = df2['sentences'].apply(len)
    
    # Remove notes with no valid sentences
    df2 = df2[df2['num_sentences'] > 0].reset_index(drop=True)
    
    print(f"After filtering: {len(df2)} notes with valid sentences")
    
    # Save the improved data
    df2.to_parquet("./reaction_sentences_improved.parquet", index=False)
    
    # Explode sentences
    df_sentences = df2.explode('sentences').reset_index()
    df_sentences.rename(columns={'index': 'note_index', 'sentences': 'sentence'}, inplace=True)
    
    # Additional safety check at sentence level
    df_sentences = df_sentences[df_sentences['sentence'].str.len() >= 5].copy()
    df_sentences = df_sentences[df_sentences['sentence'].str.contains(r'[a-zA-Z]', regex=True)].copy()
    
    # Count occurrences
    sentence_counts = df_sentences['sentence'].value_counts().reset_index()
    sentence_counts.columns = ['sentence', 'count']
    df_sentences = df_sentences.merge(sentence_counts, on='sentence', how='left')
    
    # Save improved sentences
    df_sentences.to_parquet("./sentences_improved.parquet", index=False)
    
    print(f"Final dataset: {len(df_sentences)} sentence instances")
    print(f"Unique sentences: {len(sentence_counts)}")
    
    # Show top sentences
    print("\nTop 10 most frequent sentences:")
    for i, (sentence, count) in enumerate(sentence_counts.head(10).values):
        print(f"{i+1:2d}. ({count:3d}x) {sentence[:80]}{'...' if len(sentence) > 80 else ''}")
    
    return df_sentences, sentence_counts

# Run improved processing
try:
    df_improved, counts_improved = improved_sentence_processing()
except Exception as e:
    print(f"Error in processing: {e}")
    print("Please check your file paths and data structure")

# Additional diagnostic function
def diagnose_original_notes():
    """Look at the original notes to understand the issue"""
    
    df = pd.read_parquet("./reactions.parquet")
    df2 = df[['notes']].drop_duplicates().reset_index(drop=True)
    
    print("=== ORIGINAL NOTES ANALYSIS ===")
    
    # Find very short notes
    short_notes = df2[df2['notes'].str.len() <= 10]
    print(f"Notes with 10 characters or less: {len(short_notes)}")
    
    if len(short_notes) > 0:
        print("Examples of very short notes:")
        for i, note in enumerate(short_notes['notes'].head(10)):
            print(f"  '{note}' (length: {len(note)})")
    
    # Find notes that are just punctuation
    punct_notes = df2[df2['notes'].str.match(r'^[\s\W]+$', na=False)]
    print(f"\nNotes that are just punctuation/whitespace: {len(punct_notes)}")
    
    if len(punct_notes) > 0:
        print("Examples:")
        for note in punct_notes['notes'].head(5):
            print(f"  '{note}'")
    
    # Check for empty or null notes
    empty_notes = df2[df2['notes'].isna() | (df2['notes'] == '')]
    print(f"\nEmpty or null notes: {len(empty_notes)}")
    
    return df2

# Run diagnosis
try:
    original_analysis = diagnose_original_notes()
except Exception as e:
    print(f"Error in diagnosis: {e}")