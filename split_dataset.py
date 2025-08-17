import os
import shutil
import random
from pathlib import Path

def create_split_directories(base_dir):
    """Create train, val, and test directories for each species."""
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for species in os.listdir('dataset'):
            if os.path.isdir(os.path.join('dataset', species)):
                os.makedirs(os.path.join(split_dir, species), exist_ok=True)

def split_dataset(source_dir='dataset', target_dir='split_dataset', train_ratio=0.7, val_ratio=0.15):
    """Split the dataset into training, validation, and test sets."""
    # Create split directories
    create_split_directories(target_dir)
    
    # Get all species directories
    species_dirs = [d for d in os.listdir(source_dir) 
                   if os.path.isdir(os.path.join(source_dir, d))]
    
    print(f"\nTotal number of species to process: {len(species_dirs)}")
    print("Starting dataset split...\n")
    
    # Process each species directory
    for species in species_dirs:
        species_dir = os.path.join(source_dir, species)
            
        # Get all image files
        image_files = [f for f in os.listdir(species_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"Warning: No images found in {species} directory")
            continue
            
        # Shuffle the files
        random.shuffle(image_files)
        
        # Calculate split sizes
        total_files = len(image_files)
        train_size = int(total_files * train_ratio)
        val_size = int(total_files * val_ratio)
        
        # Split the files
        train_files = image_files[:train_size]
        val_files = image_files[train_size:train_size + val_size]
        test_files = image_files[train_size + val_size:]
        
        # Copy files to their respective directories
        for split, files in [('train', train_files), 
                           ('val', val_files), 
                           ('test', test_files)]:
            for file in files:
                src = os.path.join(species_dir, file)
                dst = os.path.join(target_dir, split, species, file)
                shutil.copy2(src, dst)
        
        print(f"Processed {species}:")
        print(f"  Total images: {total_files}")
        print(f"  Training set: {len(train_files)} images")
        print(f"  Validation set: {len(val_files)} images")
        print(f"  Test set: {len(test_files)} images")
        print("-" * 50)

    # Verify the splits
    print("\nVerifying splits...")
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(target_dir, split)
        species_count = len([d for d in os.listdir(split_dir) 
                           if os.path.isdir(os.path.join(split_dir, d))])
        print(f"{split.capitalize()} set contains {species_count} species")

if __name__ == "__main__":
    split_dataset() 