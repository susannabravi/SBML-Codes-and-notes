import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def plot_data(
    data,
    logscale_x=False,
    logscale_y=False,
    outputname="fig1.png",
    output_dir="plots",
    n_bins='auto',
    discrete = None,
    boxplot=False,
    title=None,
    xlabel=None,
    ylabel="Density",
    color=None,
    cut_percentile=(0, 100),
    log_scale=None,
    save = False,
    show = True,
    figsize = (6,4)
):
    plt.style.use('ggplot')
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Apply percentile cut if needed
    if cut_percentile != (0, 100):
        lower, upper = cut_percentile
        lower_val = np.percentile(data, lower)
        upper_val = np.percentile(data, upper)
        data = np.array([x for x in data if lower_val <= x <= upper_val])

    # Log transform data if needed
    if logscale_x:
        # For integers with zeros, simply filter out zeros before applying log10
        data = np.array([x for x in data if x > 0])
        data = np.log10(data)

    # Default labels
    if title is None:
        title = "Distribution of Reaction Counts"
    if xlabel is None:
        xlabel = "Reaction Count" if not logscale_x else "log(1 + Reaction Count)"

    # Use ggplot-style color if none provided
    if color is None:
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

    # Set up plot layout
    if boxplot:
        fig, axs = plt.subplots(2, 1, figsize = figsize, gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax = plt.subplots(1, 1, figsize = figsize)
        axs = [ax]

    # Histogram
    sns.histplot(data, stat='density', kde=True, bins=n_bins, ax=axs[0], color=color, log_scale=log_scale, discrete=discrete)
    axs[0].set_title(title)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    axs[0].grid(True)

    if logscale_y:
        axs[0].set_yscale('log')

    # Optional boxplot
    if boxplot:
        sns.boxplot(data, ax=axs[1], orient='h', color=color)
        axs[1].set_xlabel(xlabel)
        axs[1].grid(True)

    # Save figure
    full_path = os.path.join(output_dir, outputname)
    plt.tight_layout()
    if save == True:
        plt.savefig(full_path)
        plt.close()
    if show == True:
        plt.show()

def print_stats(data):
    print(f"\nDescriptive Statistics:")
    #print(f"Count: {data.count()}")
    print(f"Mean: {data.mean():.2f}")
    print(f"Median: {data.median():.2f}")
    print(f"Standard Deviation: {data.std():.2f}")
    print(f"Minimum: {data.min()}")
    print(f"Maximum: {data.max()}")
    print(f"5th percentile: {np.percentile(data, 5):.2f}")
    print(f"25th percentile (Q1): {np.percentile(data, 25):.2f}")
    print(f"75th percentile (Q3): {np.percentile(data, 75):.2f}")
    print(f"95th percentile: {np.percentile(data, 95):.2f}")
    print(f"IQR (Interquartile Range): {np.percentile(data, 75) - np.percentile(data, 25):.2f}")

def extract_references(entry):
    if not isinstance(entry, str):
        return []
    refs = []
    for group in entry.split(';'):
        group = group.strip().strip('()')
        if group:
            refs.append(group)
    return refs


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
        
        # Filter out null values
        text_data = df_sample[column].dropna()
        
        if len(text_data) == 0:
            print(f"No valid text data found in column '{column}'")
            continue
        
        # Tokenize all texts
        token_lengths = []
        all_tokens = []
        
        for i, text in enumerate(text_data):
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
            decoded_tokens = [tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]
            
            print(f"\nOriginal text ({len(token_ids)} tokens):")
            print(f"{str(text)}")
            print(f"Token IDs: {token_ids}")
            print(f"Decoded tokens: {decoded_tokens}")