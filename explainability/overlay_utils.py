import cv2
import numpy as np

def create_overlay(original_image, heatmap, alpha=0.4):
    """Create overlay of original image and heatmap"""
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    if original_image.dtype != np.uint8:
        original_image = np.uint8(255 * original_image)
    
    if heatmap.shape[:2] != original_image.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    
    return overlay

def save_gradcam_visualization(original_img, heatmap, overlay, save_path_prefix):
    """Save original, heatmap, and overlay images"""
    cv2.imwrite(f"{save_path_prefix}_original.png", original_img)
    cv2.imwrite(f"{save_path_prefix}_heatmap.png", heatmap)
    cv2.imwrite(f"{save_path_prefix}_overlay.png", overlay)
