import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from tensorflow.keras.models import load_model
from preprocessing.augmentation import create_data_generators
from utils.metrics import calculate_metrics, plot_confusion_matrix
from utils.logger import setup_logger

def evaluate_mobilenetv3():
    """Evaluate MobileNetV3 model"""
    logger = setup_logger('MobileNetV3_Evaluation', 'models/mobilenetv3/evaluation.log')
    
    logger.info("Loading model...")
    model = load_model('models/mobilenetv3/weights/mobilenetv3_best.h5')
    
    logger.info("Loading test data...")
    _, _, test_gen = create_data_generators(
        'data/processed/train',
        'data/processed/val',
        'data/processed/test',
        batch_size=32
    )
    
    logger.info("Making predictions...")
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    class_names = list(test_gen.class_indices.keys())
    
    logger.info("Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, predictions, class_names)
    
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
    
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        'models/mobilenetv3/confusion_matrix.png'
    )
    
    logger.info("Evaluation completed!")
    
    return metrics

if __name__ == "__main__":
    evaluate_mobilenetv3()
