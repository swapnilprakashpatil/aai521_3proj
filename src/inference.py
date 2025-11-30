"""
Inference script for flood detection models.
Runs predictions on new images and visualizes results.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add src to path
sys.path.append(str(Path(__file__).parent))

import config
from models import create_model
from preprocessing import normalize_image


# Color map for visualization (RGB)
COLOR_MAP = {
    0: [0, 0, 0],           # Background - Black
    1: [255, 0, 0],         # Flooded Building - Red
    2: [0, 255, 0],         # Non-Flooded Building - Green
    3: [0, 0, 255],         # Flooded Road - Blue
    4: [255, 255, 0],       # Non-Flooded Road - Yellow
    5: [0, 255, 255],       # Water - Cyan
    6: [255, 0, 255]        # Tree - Magenta
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference on flood detection images')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str, required=True,
                        choices=['unet++', 'deeplabv3+', 'segformer'],
                        help='Model architecture')
    parser.add_argument('--pre-image', type=str, required=True,
                        help='Path to pre-event image')
    parser.add_argument('--post-image', type=str, required=True,
                        help='Path to post-event image')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save output visualization')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--patch-size', type=int, default=config.PATCH_SIZE,
                        help='Patch size for inference')
    
    return parser.parse_args()


def load_image(image_path):
    """Load and preprocess image."""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    return img_array


def create_input_tensor(pre_image, post_image):
    """
    Create 6-channel input tensor from pre and post event images.
    
    Args:
        pre_image: Pre-event RGB image (H, W, 3)
        post_image: Post-event RGB image (H, W, 3)
    
    Returns:
        torch.Tensor: 6-channel input tensor (1, 6, H, W)
    """
    # Ensure same size
    if pre_image.shape != post_image.shape:
        raise ValueError("Pre and post images must have the same dimensions")
    
    # Normalize images
    pre_norm = normalize_image(pre_image)
    post_norm = normalize_image(post_image)
    
    # Stack to create 6-channel image
    combined = np.concatenate([pre_norm, post_norm], axis=-1)  # (H, W, 6)
    
    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0).float()  # (1, 6, H, W)
    
    return tensor


def predict_on_patches(model, image_tensor, patch_size, device):
    """
    Run inference on image patches.
    
    Args:
        model: PyTorch model
        image_tensor: Input tensor (1, 6, H, W)
        patch_size: Size of patches
        device: Device to use
    
    Returns:
        np.ndarray: Prediction mask (H, W)
    """
    model.eval()
    
    _, _, h, w = image_tensor.shape
    
    # If image is smaller than patch size, pad it
    if h < patch_size or w < patch_size:
        pad_h = max(0, patch_size - h)
        pad_w = max(0, patch_size - w)
        image_tensor = torch.nn.functional.pad(
            image_tensor, 
            (0, pad_w, 0, pad_h), 
            mode='reflect'
        )
        _, _, h, w = image_tensor.shape
    
    # Create output mask
    prediction_mask = np.zeros((h, w), dtype=np.uint8)
    count_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Calculate number of patches
    stride = patch_size // 2  # 50% overlap
    n_patches_h = (h - patch_size) // stride + 1
    n_patches_w = (w - patch_size) // stride + 1
    
    print(f"Running inference on {n_patches_h * n_patches_w} patches...")
    
    with torch.no_grad():
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # Extract patch
                start_h = i * stride
                start_w = j * stride
                end_h = start_h + patch_size
                end_w = start_w + patch_size
                
                patch = image_tensor[:, :, start_h:end_h, start_w:end_w]
                patch = patch.to(device)
                
                # Predict
                output = model(patch)
                pred = torch.argmax(output, dim=1).cpu().numpy()[0]
                
                # Accumulate predictions
                prediction_mask[start_h:end_h, start_w:end_w] += pred
                count_mask[start_h:end_h, start_w:end_w] += 1
    
    # Average overlapping predictions
    prediction_mask = (prediction_mask / count_mask).astype(np.uint8)
    
    return prediction_mask


def mask_to_rgb(mask):
    """Convert prediction mask to RGB image."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in COLOR_MAP.items():
        rgb[mask == class_id] = color
    
    return rgb


def visualize_results(pre_image, post_image, prediction_mask, output_path):
    """Create visualization of results."""
    # Convert mask to RGB
    pred_rgb = mask_to_rgb(prediction_mask)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Pre-event image
    axes[0, 0].imshow(pre_image)
    axes[0, 0].set_title('Pre-Event Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Post-event image
    axes[0, 1].imshow(post_image)
    axes[0, 1].set_title('Post-Event Image', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Prediction mask
    axes[1, 0].imshow(pred_rgb)
    axes[1, 0].set_title('Prediction Mask', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Overlay on post-event
    overlay = post_image.copy()
    overlay = overlay.astype(np.float32)
    pred_rgb_float = pred_rgb.astype(np.float32)
    overlay = 0.6 * overlay + 0.4 * pred_rgb_float
    overlay = overlay.astype(np.uint8)
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay on Post-Event', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor=np.array(COLOR_MAP[i])/255, 
                      label=config.CLASS_NAMES[i])
        for i in range(config.NUM_CLASSES)
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=4, fontsize=12, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")


def print_statistics(prediction_mask):
    """Print prediction statistics."""
    print("\nPrediction Statistics:")
    print("-" * 60)
    
    total_pixels = prediction_mask.size
    
    for class_id, class_name in enumerate(config.CLASS_NAMES):
        count = np.sum(prediction_mask == class_id)
        percentage = (count / total_pixels) * 100
        print(f"{class_name:<30} {count:>10} pixels ({percentage:>5.2f}%)")
    
    print("-" * 60)


def main():
    """Main inference function."""
    args = parse_args()
    
    print("="*80)
    print("FLOOD DETECTION INFERENCE")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Pre-event image: {args.pre_image}")
    print(f"Post-event image: {args.post_image}")
    print(f"Device: {args.device}\n")
    
    # Load images
    print("Loading images...")
    pre_image = load_image(args.pre_image)
    post_image = load_image(args.post_image)
    
    print(f"Image size: {pre_image.shape}")
    
    # Create input tensor
    print("Creating input tensor...")
    input_tensor = create_input_tensor(pre_image, post_image)
    
    # Create model
    print(f"Creating {args.model} model...")
    model = create_model(
        args.model,
        num_classes=config.NUM_CLASSES,
        in_channels=6,
        **config.MODEL_CONFIGS.get(args.model, {})
    )
    
    # Load checkpoint
    print(f"Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    
    # Run inference
    print("Running inference...")
    prediction_mask = predict_on_patches(
        model=model,
        image_tensor=input_tensor,
        patch_size=args.patch_size,
        device=args.device
    )
    
    # Print statistics
    print_statistics(prediction_mask)
    
    # Visualize results
    print("\nCreating visualization...")
    visualize_results(pre_image, post_image, prediction_mask, args.output)
    
    # Save prediction mask
    mask_output = Path(args.output).parent / f"{Path(args.output).stem}_mask.npy"
    np.save(mask_output, prediction_mask)
    print(f"Prediction mask saved to: {mask_output}")
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
