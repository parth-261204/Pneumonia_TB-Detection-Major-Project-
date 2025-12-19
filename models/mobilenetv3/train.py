import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import yaml
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from models.mobilenetv3.model import create_mobilenetv3_model, unfreeze_and_finetune
from preprocessing.augmentation import create_data_generators
from utils.logger import setup_logger
from utils.metrics import plot_training_history

def train_mobilenetv3():
    """Train MobileNetV3 model"""
    logger = setup_logger('MobileNetV3_Training', 'models/mobilenetv3/training.log')
    
    with open('config/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Loading data generators...")
    train_gen, val_gen, _ = create_data_generators(
        'data/processed/train',
        'data/processed/val',
        'data/processed/test',
        batch_size=config['training']['batch_size'],
        target_size=tuple(config['training']['image_size'])
    )
    
    logger.info("Creating MobileNetV3 model...")
    model = create_mobilenetv3_model(
        input_shape=(224, 224, 3),
        num_classes=3,
        learning_rate=config['training']['learning_rate']
    )
    
    logger.info(f"Model created with {model.count_params()} parameters")
    
    os.makedirs('models/mobilenetv3/weights', exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            'models/mobilenetv3/weights/mobilenetv3_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config['callbacks']['early_stopping']['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config['callbacks']['reduce_lr']['factor'],
            patience=config['callbacks']['reduce_lr']['patience'],
            min_lr=config['callbacks']['reduce_lr']['min_lr'],
            verbose=1
        )
    ]
    
    logger.info("Starting initial training (frozen base)...")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Fine-tuning model (unfrozen layers)...")
    model = unfreeze_and_finetune(model, learning_rate=0.00001)
    
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config['training']['epochs'] - 20,
        callbacks=callbacks,
        verbose=1
    )
    
    history1.history['accuracy'].extend(history2.history['accuracy'])
    history1.history['val_accuracy'].extend(history2.history['val_accuracy'])
    history1.history['loss'].extend(history2.history['loss'])
    history1.history['val_loss'].extend(history2.history['val_loss'])
    
    plot_training_history(history1, 'models/mobilenetv3/training_history.png')
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {max(history1.history['val_accuracy']):.4f}")
    
    return model

if __name__ == "__main__":
    train_mobilenetv3()
