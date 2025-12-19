import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
from ensemble.ensemble_model import EnsembleModel
from utils.logger import setup_logger

def predict_single_xray(image_path, model_type='ensemble'):
    """Predict diagnosis for single X-ray image"""
    logger = setup_logger('Single_Prediction')
    
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return None
    
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    
    if model_type == 'ensemble':
        ensemble = EnsembleModel.load_from_config()
        result = ensemble.predict(img_normalized)
    else:
        from tensorflow.keras.models import load_model
        model = load_model(model_type)
        predictions = model.predict(np.expand_dims(img_normalized, axis=0), verbose=0)
        pred_class = np.argmax(predictions[0])
        class_names = ['Normal', 'Tuberculosis', 'Pneumonia']
        
        result = {
            'predicted_class': class_names[pred_class],
            'confidence': float(predictions[0][pred_class]),
            'probabilities': {
                class_names[i]: float(predictions[0][i]) 
                for i in range(len(class_names))
            }
        }
    
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict TB/Pneumonia from X-ray')
    parser.add_argument('image_path', type=str, help='Path to X-ray image')
    parser.add_argument('--model', type=str, default='ensemble', help='Model type or path')
    
    args = parser.parse_args()
    
    result = predict_single_xray(args.image_path, args.model)
    
    if result:
        print("\n=== Prediction Results ===")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print("\nClass Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob*100:.2f}%")
