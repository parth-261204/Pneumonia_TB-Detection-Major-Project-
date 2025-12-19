import cv2
import numpy as np

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return enhanced_rgb

def batch_clahe_preprocessing(images, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE to batch of images"""
    enhanced_images = []
    
    for img in images:
        enhanced = apply_clahe(img, clip_limit, tile_grid_size)
        enhanced_images.append(enhanced)
    
    return np.array(enhanced_images)
