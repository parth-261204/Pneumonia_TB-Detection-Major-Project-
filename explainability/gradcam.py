import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap for given image"""
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def get_last_conv_layer_name(model):
    """Find the last convolutional layer in model"""
    conv_layer_names = []
    
    for layer in model.layers:
        if 'conv' in layer.name.lower():
            conv_layer_names.append(layer.name)
    
    if not conv_layer_names:
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                for sublayer in reversed(layer.layers):
                    if 'conv' in sublayer.name.lower():
                        return sublayer.name
    
    return conv_layer_names[-1] if conv_layer_names else None

def generate_gradcam(model, image, target_size=(224, 224), pred_index=None):
    """Generate Grad-CAM heatmap for a preprocessed image"""
    if len(image.shape) == 3:
        img_array = np.expand_dims(image, axis=0)
    else:
        img_array = image
    
    last_conv_layer = get_last_conv_layer_name(model)
    
    if last_conv_layer is None:
        raise ValueError("Could not find convolutional layer in model")
    
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, pred_index)
    
    heatmap = cv2.resize(heatmap, target_size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap
