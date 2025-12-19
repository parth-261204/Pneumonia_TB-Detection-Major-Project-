import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

def create_directory_structure(base_path):
    """Create processed data directory structure"""
    splits = ['train', 'val', 'test']
    classes = ['normal', 'tuberculosis', 'pneumonia']
    
    for split in splits:
        for cls in classes:
            path = os.path.join(base_path, 'processed', split, cls)
            os.makedirs(path, exist_ok=True)
    
    print(f"Created directory structure at {base_path}/processed")

def resize_and_normalize(image_path, target_size=(224, 224)):
    """Resize and normalize image"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return None
    
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = cv2.equalizeHist(img)
    
    return img

def process_dataset(raw_path, processed_path, target_size=(224, 224), split_ratios=(0.7, 0.15, 0.15)):
    """Process raw images and split into train/val/test"""
    create_directory_structure(os.path.dirname(processed_path))
    
    classes = ['normal', 'tuberculosis', 'pneumonia']
    
    for cls in classes:
        cls_path = os.path.join(raw_path, cls)
        
        if not os.path.exists(cls_path):
            print(f"Warning: Class folder {cls} not found")
            continue
        
        images = [f for f in os.listdir(cls_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        np.random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])
        
        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }
        
        for split, split_images in splits.items():
            print(f"\nProcessing {cls} - {split}: {len(split_images)} images")
            
            for img_name in tqdm(split_images):
                src_path = os.path.join(cls_path, img_name)
                processed_img = resize_and_normalize(src_path, target_size)
                
                if processed_img is not None:
                    dst_path = os.path.join(processed_path, split, cls, img_name)
                    cv2.imwrite(dst_path, processed_img)
        
        print(f"Completed processing {cls}")

if __name__ == "__main__":
    raw_path = "data/raw"
    processed_path = "data/processed"
    
    process_dataset(raw_path, processed_path, target_size=(224, 224))
    print("\nDataset preprocessing completed!")
