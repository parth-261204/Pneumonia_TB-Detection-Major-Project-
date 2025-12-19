import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from explainability.gradcam import generate_gradcam
from explainability.overlay_utils import create_overlay, save_gradcam_visualization
from utils.logger import setup_logger

def generate_heatmaps_for_model(model_path, image_path, class_names, output_dir):
    """Generate Grad-CAM heatmaps for a specific model"""
    logger = setup_logger('Heatmap_Generator')
    
    logger.info(f"Loading model: {model_path}")
    model = load_model(model_path)
    
    logger.info(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    
    predictions = model.predict(np.expand_dims(img_normalized, axis=0))
    pred_class = np.argmax(predictions[0])
    confidence = predictions[0][pred_class]
    
    logger.info(f"Prediction: {class_names[pred_class]} ({confidence*100:.2f}%)")
    
    heatmap = generate_gradcam(model, img_normalized, pred_index=pred_class)
    overlay = create_overlay(img_resized, heatmap, alpha=0.4)
    
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.basename(model_path).replace('.h5', '')
    save_path = os.path.join(output_dir, model_name)
    
    save_gradcam_visualization(img_resized, heatmap, overlay, save_path)
    
    logger.info(f"Saved visualizations to {output_dir}")
    
    return {
        'prediction': class_names[pred_class],
        'confidence': float(confidence),
        'heatmap': heatmap,
        'overlay': overlay
    }

if __name__ == "__main__":
    class_names = ['Normal', 'Tuberculosis', 'Pneumonia']
    models = [
        'models/mobilenetv3/weights/mobilenetv3_best.h5',
        'models/efficientnetb0/weights/efficientnetb0_best.h5',
        'models/densenet121/weights/densenet121_best.h5',
        'models/resnet50/weights/resnet50_best.h5'
    ]
    
    image_path = 'path/to/test/image.png'
    
    for model_path in models:
        if os.path.exists(model_path):
            generate_heatmaps_for_model(
                model_path, 
                image_path, 
                class_names, 
                'reports/images/gradcam'
            )
