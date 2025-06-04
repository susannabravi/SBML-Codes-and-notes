import pandas as pd
import pysbd

# Load and clean original data
df = pd.read_parquet("./reactions.parquet")
df = df.drop_duplicates(subset=[col for col in df.columns if col != 'file_id']).reset_index(drop=True)
df = df.drop_duplicates().reset_index(drop=True)

# Extract and deduplicate notes
df2 = df[['notes','original_notes']].drop_duplicates().reset_index(drop=True)

# Initialize sentence segmenter
segmenter = pysbd.Segmenter(language="en", clean=True)

# Split each note into sentences
df2['sentences'] = df2['notes'].apply(lambda text: segmenter.segment(text))

df2['num_sentences'] = df2['sentences'].apply(len)

# Save data
df2.to_parquet("./reaction_sentences.parquet", index=False)

# Explode so each sentence becomes a separate row
df_sentences = df2.explode('sentences').reset_index()
# After exploding and before counting
df_sentences['sentences'] = df_sentences['sentences'].str.strip()

'''
# Drop any sentence that is empty or just punctuation
df_sentences = df_sentences[
    df_sentences['sentences'].str.len() > 2  # remove very short strings
]
df_sentences = df_sentences[
    ~df_sentences['sentences'].isin([".", ",", ";", ":", "!", "?", "()", "[]", "{}"])
]
'''

df_sentences.rename(columns={'index': 'note_index', 'sentences': 'sentence'}, inplace=True)

# Count occurrences of each sentence
sentence_counts = df_sentences['sentence'].value_counts().reset_index()
sentence_counts.columns = ['sentence', 'count']

# Merge sentence counts back into the exploded DataFrame
df_sentences = df_sentences.merge(sentence_counts, on='sentence', how='left')

# Save to file
df_sentences.to_parquet("./sentences.parquet", index=False)
