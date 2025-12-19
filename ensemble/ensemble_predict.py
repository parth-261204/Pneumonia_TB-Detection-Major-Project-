import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import cv2
from ensemble.ensemble_model import EnsembleModel
from explainability.gradcam import generate_gradcam
from explainability.overlay_utils import create_overlay
from utils.logger import setup_logger

def ensemble_predict_with_gradcam(image_path, ensemble_model, output_dir='reports/images'):
    """Perform ensemble prediction with Grad-CAM visualization"""
    logger = setup_logger('Ensemble_Prediction')
    
    logger.info(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    
    logger.info("Making ensemble prediction...")
    result = ensemble_model.predict(img_normalized)
    
    logger.info(f"Prediction: {result['predicted_class']} ({result['confidence']*100:.2f}%)")
    
    best_model_idx = np.argmax([
        result['individual_predictions'][name]['confidence'] 
        for name in ensemble_model.model_names
    ])
    best_model = ensemble_model.models[best_model_idx]
    
    logger.info("Generating Grad-CAM heatmap...")
    heatmap = generate_gradcam(
        best_model, 
        img_normalized, 
        pred_index=result['predicted_index']
    )
    overlay = create_overlay(img_resized, heatmap, alpha=0.4)
    
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, 'original', 'input.png'), img_resized)
    cv2.imwrite(os.path.join(output_dir, 'gradcam', 'heatmap.png'), heatmap)
    cv2.imwrite(os.path.join(output_dir, 'overlays', 'overlay.png'), overlay)
    
    result['heatmap'] = heatmap
    result['overlay'] = overlay
    result['original'] = img_resized
    
    return result

if __name__ == "__main__":
    ensemble = EnsembleModel.load_from_config()
    
    image_path = 'path/to/test/image.png'
    result = ensemble_predict_with_gradcam(image_path, ensemble)
    
    print("\n=== Ensemble Prediction Results ===")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print("\nClass Probabilities:")
    for cls, prob in result['probabilities'].items():
        print(f"  {cls}: {prob*100:.2f}%")
