import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

def create_mobilenetv3_model(input_shape=(224, 224, 3), num_classes=3, learning_rate=0.0001):
    """Create MobileNetV3 transfer learning model"""
    base_model = MobileNetV3Small(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.3, name='dropout')(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.2, name='dropout2')(x)
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def unfreeze_and_finetune(model, learning_rate=0.00001):
    """Unfreeze base model for fine-tuning"""
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers[-30:]:
                sublayer.trainable = True
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
