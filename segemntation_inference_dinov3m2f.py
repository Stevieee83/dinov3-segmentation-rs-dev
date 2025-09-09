from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import colormaps
from functools import partial
from dinov3.eval.segmentation.inference import make_inference
import os
import requests

from load_data_seg_1_image import LoadData

import wandb

import argparse

# Defines the ArgumentParser object
parser = argparse.ArgumentParser()

# Input parameters
parser.add_srgument("--img_size", type=int, default=896)
parser.add_srgument("--run", type=int, default=1)
parser.add_argument("--image_dir", type=str, default="/uoa/scratch/users/r02sw23/borebreen-drone-image-data/images/borebreen_crop_drone_11.png")
parser.add_argument("--labels_dir", type=str, default="/uoa/scratch/users/r02sw23/borebreen-drone-image-data/masks/borebreen_crop_drone_11.png")
parser.add_argument("--output_dir", type=str, default='/uoa/scratch/users/r02sw23/dinov3-main-fedr/segmentation_head_results/ADE20K_m2f')
parser.add_argument("--backbone_weights", type=str, default='/uoa/scratch/users/r02sw23/dinov3-main/pre_trained_weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth')
parser.add_argument("--weights", type=str, default='/uoa/scratch/users/r02sw23/dinov3-main/pre_trained_weights/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth')
parser.add_argument("--REPO_DIR", type=str, default='/uoa/scratch/users/r02sw23/dinov3-main-fedr')

# Main function to sequence the script
def main():
    # Creates the ArgumentParser object in the main function
    args = parser.parse_args()

    load_data = LoadData(args.image_dir, args.labels_dir)

    # Initialise the Weights and Biases run
    wandb.init(project=f"DINOv3 Segmentation ADE20K Head {args.model_type}",
                name=f"Code Run {args.run}")

    # Copy your config
    config = wandb.config
  
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading DINOv3 segmentation model...")
    try:
        segmentor = torch.hub.load(
            REPO_DIR, 
            'dinov3_vit7b16_ms', 
            source="local", 
            weights=args.weights, 
            backbone_weights=args.backbone_weights
        )
        segmentor = segmentor.to(device)
        segmentor.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get and preprocess image
    print("Getting test image...")
    # Load the image data and their labels into the CUDA runtime
    img, labels = load_data.sequence_data_loading()
    transform = make_transform(args.img_size)
    
    # Inference
    print("Running inference...")
    with torch.inference_mode():
        with torch.autocast(device.type, dtype=torch.bfloat16):
            batch_img = transform(img)[None].to(device)
            
            # Raw predictions
            pred_vit7b = segmentor(batch_img)
            print(f"Raw prediction shape: {pred_vit7b.shape}")
            
            # Generate segmentation map
            segmentation_map_vit7b = make_inference(
                batch_img,
                segmentor,
                inference_mode="slide",
                decoder_head_type="m2f",
                rescale_to=img.size,  # Fixed: img.size is (width, height)
                n_output_channels=150,
                crop_size=(args.img_size, args.img_size),
                stride=(args.img_size, args.img_size),
                output_activation=partial(torch.nn.functional.softmax, dim=1),
            ).argmax(dim=1, keepdim=True)
            
            print(f"Segmentation map shape: {segmentation_map_vit7b.shape}")
    
    # Visualization
    print("Creating visualization...")
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(122)
    plt.imshow(segmentation_map_vit7b[0, 0].cpu().numpy(), cmap=colormaps["Spectral"])
    plt.title("Segmentation Map")
    plt.axis("off")
    
    # Save results
    output_path = os.path.join(args.output_dir, 'subplot_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Results saved to: {output_path}")
    
    # Also save individual images
    img.save(os.path.join(args.output_dir, 'original_image.png'))
    
    # Save segmentation map as image
    seg_img = Image.fromarray(
        (segmentation_map_vit7b[0, 0].cpu().numpy() * 255 / 150).astype('uint8')
    )
    seg_img.save(os.path.join(args.output_dir, 'segmentation_map.png'))
    
    print("Processing complete!")

    # Close your Weights and biases run
    wandb.finish()

def make_transform(resize_size: int = 768):
    """Create image preprocessing transform."""
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])

if __name__ == "__main__":
    main()
