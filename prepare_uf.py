import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from collections import defaultdict
import hashlib
from tqdm import tqdm
import random

def deduplicate_by_prompt(dataset, prompt_column='prompt'):
    """
    Deduplicate dataset by prompt, keeping the first occurrence of each unique prompt
    """
    print(f"Original dataset size: {len(dataset)}")
    
    # Get all prompts
    prompts = dataset[prompt_column]
    
    # Track seen prompts and their indices
    seen_prompts = set()
    unique_indices = []
    
    for i, prompt in enumerate(tqdm(prompts, desc="Deduplicating")):
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            unique_indices.append(i)
    
    print(f"After deduplication: {len(unique_indices)}")
    print(f"Removed {len(dataset) - len(unique_indices)} duplicate prompts")
    
    return dataset.select(unique_indices)

def split_dataset(dataset, test_size=3000, val_size=2000):
    """
    Split dataset into train/validation/test sets
    """
    total_size = len(dataset)
    print(f"Total dataset size: {total_size}")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Shuffle indices
    indices = list(range(total_size))
    random.shuffle(indices)
    
    # Split indices
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]
    
    print(f"Test set size: {len(test_indices)}")
    print(f"Validation set size: {len(val_indices)}")
    print(f"Training set size: {len(train_indices)}")
    
    # Create datasets
    test_dataset = dataset.select(test_indices)
    val_dataset = dataset.select(val_indices)
    train_dataset = dataset.select(train_indices)
    
    return train_dataset, val_dataset, test_dataset

def main():
    print("Loading ultrafeedback_binarized dataset...")
    
    # Load the dataset
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    
    print("Dataset loaded successfully!")
    print(f"Original train_prefs size: {len(ds)}")
    
    # Also load test_prefs to get additional samples for test set
    ds_test_orig = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
    print(f"Original test_prefs size: {len(ds_test_orig)}")
    
    # Combine train and test for processing
    combined_ds = ds
    
    # Deduplicate by prompt
    print("\nDeduplicating by prompt...")
    deduped_ds = deduplicate_by_prompt(combined_ds)
    
    # Check if we need to add samples from test_prefs to reach 3000 test samples
    total_available = len(deduped_ds)
    needed_for_test = 3000
    needed_for_val = 2000
    
    if total_available < needed_for_test + needed_for_val:
        print(f"Warning: Total available samples ({total_available}) is less than required (test: {needed_for_test}, val: {needed_for_val})")
        # Adjust sizes proportionally
        if total_available > needed_for_val:
            test_size = min(needed_for_test, total_available - needed_for_val)
            val_size = needed_for_val
        else:
            test_size = total_available // 2
            val_size = total_available - test_size
    else:
        test_size = needed_for_test
        val_size = needed_for_val
    
    # Split dataset
    print(f"\nSplitting dataset into train/val/test...")
    train_ds, val_ds, test_ds = split_dataset(deduped_ds, test_size=test_size, val_size=val_size)
    
    # Create final dataset dict
    processed_dataset = DatasetDict({
        'train': train_ds,
        'validation': val_ds,
        'test': test_ds
    })
    
    print(f"\nFinal dataset sizes:")
    print(f"Train: {len(processed_dataset['train'])}")
    print(f"Validation: {len(processed_dataset['validation'])}")
    print(f"Test: {len(processed_dataset['test'])}")
    
    # Upload to Hugging Face Hub
    print(f"\nUploading to Hugging Face Hub...")
    repo_name = "weqweasdas/ultrafeedback_binarized_processed"
    
    try:
        processed_dataset.push_to_hub(repo_name, private=False)
        print(f"Successfully uploaded to {repo_name}")
    except Exception as e:
        print(f"Error uploading to Hub: {e}")
        print("Saving locally instead...")
        processed_dataset.save_to_disk("./ultrafeedback_binarized_processed")
        print("Saved locally to ./ultrafeedback_binarized_processed")
    
    return processed_dataset

if __name__ == "__main__":
    processed_ds = main()
