import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ensemble.ensemble_model import EnsembleModel
from utils.logger import setup_logger

def predict_batch(image_dir, output_csv='reports/batch_predictions.csv'):
    """Predict diagnosis for batch of X-ray images"""
    logger = setup_logger('Batch_Prediction')
    
    ensemble = EnsembleModel.load_from_config()
    
    image_files = [f for f in os.listdir(image_dir) 
                   if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    results = []
    
    logger.info(f"Processing {len(image_files)} images...")
    
    for img_file in tqdm(image_files):
        img_path = os.path.join(image_dir, img_file)
        
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized / 255.0
        
        prediction = ensemble.predict(img_normalized)
        
        results.append({
            'filename': img_file,
            'predicted_class': prediction['predicted_class'],
            'confidence': prediction['confidence'],
            'normal_prob': prediction['probabilities']['Normal'],
            'tuberculosis_prob': prediction['probabilities']['Tuberculosis'],
            'pneumonia_prob': prediction['probabilities']['Pneumonia']
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    logger.info(f"Results saved to {output_csv}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch predict TB/Pneumonia')
    parser.add_argument('image_dir', type=str, help='Directory containing X-ray images')
    parser.add_argument('--output', type=str, default='reports/batch_predictions.csv')
    
    args = parser.parse_args()
    
    results = predict_batch(args.image_dir, args.output)
    print(f"\nProcessed {len(results)} images")
    print(f"Results saved to {args.output}")
