#!/usr/bin/env python3
"""
Simple script to create small subsets of training/validation data for testing.
"""

import json
import sys

def create_subset(input_file, output_file, num_samples=100):
    """Take first N samples from a JSON/JSONL file and save to a new file."""
    
    # Load data
    print(f"Loading {input_file}...")
    data = []
    
    with open(input_file, 'r') as f:
        try:
            # Try to load as regular JSON array
            data = json.load(f)
        except json.JSONDecodeError:
            # If that fails, try loading as JSONL (one JSON per line)
            f.seek(0)  # Go back to start of file
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    data.append(json.loads(line))
    
    # Take subset
    subset = data[:num_samples]
    
    # Save subset as regular JSON array
    with open(output_file, 'w') as f:
        json.dump(subset, f, indent=2)
    
    print(f"✓ Created {output_file} with {len(subset)} samples (from {len(data)} total)")

# Create small subsets
if __name__ == "__main__":
    # You can change these numbers to whatever you want
    TRAIN_SAMPLES = 100  # How many training samples to keep
    VAL_SAMPLES = 20     # How many validation samples to keep
    
    # Create training subset
    create_subset(
        input_file="./finetune_data/train_data.json",  # Changed from test_data.json
        output_file="./finetune_data/train_small.json",
        num_samples=TRAIN_SAMPLES
    )
    
    # Create validation subset  
    create_subset(
        input_file="./finetune_data/val_data.json", 
        output_file="./finetune_data/val_small.json",
        num_samples=VAL_SAMPLES
    )
    
    print("\nDone! Now run:")