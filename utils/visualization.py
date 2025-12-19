import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def plot_sample_images(images, labels, class_names, num_samples=9, save_path=None):
    """Plot sample images from dataset"""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f'Class: {class_names[labels[i]]}', fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def visualize_predictions(image, true_label, pred_label, confidence, class_names, save_path=None):
    """Visualize single prediction with confidence"""
    plt.figure(figsize=(8, 6))
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')
    
    color = 'green' if true_label == pred_label else 'red'
    
    title = f"True: {class_names[true_label]}\n"
    title += f"Predicted: {class_names[pred_label]} ({confidence*100:.2f}%)"
    
    plt.title(title, fontsize=14, fontweight='bold', color=color)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def create_comparison_plot(original, heatmap, overlay, prediction_text, save_path=None):
    """Create side-by-side comparison of original, heatmap, and overlay"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original X-Ray', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay with Prediction', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    fig.text(0.5, 0.02, prediction_text, ha='center', fontsize=12, 
             fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
