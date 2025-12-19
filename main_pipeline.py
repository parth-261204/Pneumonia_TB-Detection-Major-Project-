import os
import sys
import argparse
import cv2
import numpy as np
from ensemble.ensemble_model import EnsembleModel
from explainability.gradcam import generate_gradcam
from explainability.overlay_utils import create_overlay
from reports.final_report_generator import generate_diagnostic_report
from utils.logger import setup_logger

def main_pipeline(image_path, patient_info=None, output_dir='reports'):
    """
    Complete pipeline: Input X-ray -> Preprocessing -> Ensemble Prediction -> 
    Grad-CAM -> Report Generation
    """
    logger = setup_logger('Main_Pipeline', f'{output_dir}/pipeline.log')
    
    logger.info("="*60)
    logger.info("TB & PNEUMONIA DETECTION PIPELINE")
    logger.info("="*60)
    
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return None
    
    logger.info(f"Step 1: Loading and preprocessing image: {image_path}")
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    
    os.makedirs(f'{output_dir}/images/original', exist_ok=True)
    cv2.imwrite(f'{output_dir}/images/original/input.png', img_resized)
    
    logger.info("Step 2: Loading ensemble model...")
    ensemble = EnsembleModel.load_from_config()
    
    logger.info("Step 3: Making ensemble prediction...")
    prediction_result = ensemble.predict(img_normalized)
    
    logger.info(f"Prediction: {prediction_result['predicted_class']}")
    logger.info(f"Confidence: {prediction_result['confidence']*100:.2f}%")
    
    logger.info("Step 4: Generating Grad-CAM visualization...")
    best_model_idx = np.argmax([
        prediction_result['individual_predictions'][name]['confidence'] 
        for name in ensemble.model_names
    ])
    best_model = ensemble.models[best_model_idx]
    
    heatmap = generate_gradcam(
        best_model, 
        img_normalized, 
        pred_index=prediction_result['predicted_index']
    )
    
    overlay = create_overlay(img_resized, heatmap, alpha=0.4)
    
    os.makedirs(f'{output_dir}/images/gradcam', exist_ok=True)
    os.makedirs(f'{output_dir}/images/overlays', exist_ok=True)
    
    cv2.imwrite(f'{output_dir}/images/gradcam/heatmap.png', heatmap)
    cv2.imwrite(f'{output_dir}/images/overlays/overlay.png', overlay)
    
    logger.info("Step 5: Generating diagnostic report...")
    
    image_paths = {
        'original': f'{output_dir}/images/original/input.png',
        'gradcam': f'{output_dir}/images/gradcam/heatmap.png',
        'overlay': f'{output_dir}/images/overlays/overlay.png'
    }
    
    report_path = f'{output_dir}/diagnostic_report.pdf'
    generate_diagnostic_report(
        prediction_result, 
        image_paths, 
        patient_info, 
        report_path
    )
    
    logger.info("="*60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    logger.info(f"Diagnostic Report: {report_path}")
    logger.info(f"Visualizations: {output_dir}/images/")
    
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"Predicted Class: {prediction_result['predicted_class']}")
    print(f"Confidence: {prediction_result['confidence']*100:.2f}%")
    print("\nClass Probabilities:")
    for cls, prob in prediction_result['probabilities'].items():
        print(f"  {cls}: {prob*100:.2f}%")
    print("\nIndividual Model Predictions:")
    for model_name, pred in prediction_result['individual_predictions'].items():
        print(f"  {model_name}: {pred['class']} ({pred['confidence']*100:.2f}%)")
    print("="*60)
    print(f"\nFull report saved to: {report_path}")
    
    return prediction_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Complete TB & Pneumonia Detection Pipeline'
    )
    parser.add_argument(
        'image_path', 
        type=str, 
        help='Path to chest X-ray image'
    )
    parser.add_argument(
        '--patient-id', 
        type=str, 
        default='UNKNOWN',
        help='Patient ID'
    )
    parser.add_argument(
        '--age', 
        type=str, 
        default='N/A',
        help='Patient age'
    )
    parser.add_argument(
        '--gender', 
        type=str, 
        default='N/A',
        help='Patient gender'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='reports',
        help='Output directory for reports'
    )
    
    args = parser.parse_args()
    
    patient_info = {
        'id': args.patient_id,
        'age': args.age,
        'gender': args.gender,
        'date': __import__('datetime').datetime.now().strftime('%Y-%m-%d')
    }
    
    main_pipeline(args.image_path, patient_info, args.output_dir)
