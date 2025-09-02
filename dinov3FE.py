import io
import os
import pickle
import tarfile
import urllib

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
import pandas as pd

from load_data_1 import LoadData
from output_data_fe import OutputDataFE

import wandb

import argparse

# Defines the ArgumentParser object
parser = argparse.ArgumentParser()

# Input parameters
parser.add_argument("--model_type", type=str, default='ViT7b')
parser.add_argument("--run", type=int, default=1)
parser.add_argument("--image_dir", type=str, default="/uoa/scratch/users/r02sw23/borebreen-drone-image-data/images/borebreen_crop_drone_11.png")
parser.add_argument("--labels_dir", type=str, default="/uoa/scratch/users/r02sw23/borebreen-drone-image-data/masks/borebreen_crop_drone_11.png")
parser.add_argument("--output_csv", type=str, default="/uoa/scratch/users/r02sw23/dinov3-main-fedr/output_csv/")
# ------------------------------------------------------------------------

# Main funciton to sequence the Python script source code
def main():
    # Creates the ArgumentParser object in the main function
    args = parser.parse_args()
    
    load_data = LoadData(args.image_dir, args.labels_dir)

    # Initialise the Weights and Biases run
    wandb.init(project=f"DINOv3 Segmentation FE {args.model_type}",
                name=f"Code Test {args.run}")

    # Copy your config
    config = wandb.config

    DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"

    if os.getenv("DINOV3_LOCATION") is not None:
        DINOV3_LOCATION = os.getenv("DINOV3_LOCATION")
    else:
        DINOV3_LOCATION = DINOV3_GITHUB_LOCATION

    print(f"DINOv3 location set to {DINOV3_LOCATION}")

    ##############################################################################################
    # Load the DINOv3 model backbone and send to the CUDA device
    # examples of available DINOv3 models:
    MODEL_DINOV3_VITS = "dinov3_vits16"
    MODEL_DINOV3_VITSP = "dinov3_vits16plus"
    MODEL_DINOV3_VITB = "dinov3_vitb16"
    MODEL_DINOV3_VITL = "dinov3_vitl16"
    MODEL_DINOV3_VITHP = "dinov3_vith16plus"
    MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

    MODEL_NAME = MODEL_DINOV3_VIT7B

    model = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=MODEL_NAME,
        source="local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github",
        weights='/uoa/scratch/users/r02sw23/dinov3-main/pre_trained_weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth',
    )

    model.cuda()
    print(model)

    print_model_architecture(model, model_name=MODEL_NAME)

    ##############################################################################################
    # Load the image data and their labels into the CUDA runtime
    images, labels = load_data.sequence_data_loading()   

    n_images = len(images)
    assert n_images == len(labels), f"{len(images)=}, {len(labels)=}"

    ##############################################################################################
    # Building the Per-Patch Label Map
    PATCH_SIZE = 16
    IMAGE_SIZE = 768

    # quantization filter for the given patch size
    patch_quant_filter = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
    patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))

    # image resize transform to dimensions divisible by patch size
    def resize_transform(
        mask_image: Image,
        image_size: int = IMAGE_SIZE,
        patch_size: int = PATCH_SIZE,
)   -> torch.Tensor:
        w, h = mask_image.size
        h_patches = int(image_size / patch_size)
        w_patches = int((w * image_size) / (h * patch_size))
        return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))
   
    ##############################################################################################
    xs = []
    ys = []
    image_index = []

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    MODEL_TO_NUM_LAYERS = {
        MODEL_DINOV3_VITS: 12,
        MODEL_DINOV3_VITSP: 12,
        MODEL_DINOV3_VITB: 12,
        MODEL_DINOV3_VITL: 24,
        MODEL_DINOV3_VITHP: 32,
        MODEL_DINOV3_VIT7B: 40,
    }

    n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]

    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i in tqdm(range(n_images), desc="Processing images"):
                # Loading the ground truth
                mask_i = labels[i]#.split()[-1]
                mask_i_resized = resize_transform(mask_i)
                mask_i_quantized = patch_quant_filter(mask_i_resized).squeeze().view(-1).detach().cpu()
                ys.append(mask_i_quantized)
                # Loading the image data 
                image_i = images[i].convert('RGB')
                image_i_resized = resize_transform(image_i)
                image_i_resized = TF.normalize(image_i_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
                image_i_resized = image_i_resized.unsqueeze(0).cuda()

                feats = model.get_intermediate_layers(image_i_resized, n=range(n_layers), reshape=True, norm=True)
                dim = feats[-1].shape[1]
                xs.append(feats[-1].squeeze().view(dim, -1).permute(1,0).detach().cpu())

                image_index.append(i * torch.ones(ys[-1].shape))


    # Concatenate all lists into torch tensors 
    xs = torch.cat(xs)
    ys = torch.cat(ys)
    image_index = torch.cat(image_index)

    print("Design full matrix of size : ", xs.shape)
    print("Label full matrix of size : ", ys.shape)
    print("Image full index matrix of size : ", image_index.shape)

    os.makedirs(args.output_csv, exist_ok=True)
    x_csv_path = args.output_csv + 'X.csv'
    y_csv_path = args.output_csv + 'y.csv'

    # Define the OutputDataFE Python object and load the X (features) and y (labels)
    output_data = OutputDataFE(xs, ys, x_csv_path, y_csv_path)

    # Call the Python class methods to output the X and y data from the DINOv3 feature extractor
    x_df = output_data.tensor_to_df_features()
    y_df = output_data.tensor_to_df_labels()
    print('Saved xs.csv and ys.csv to their output filepath directory')

    # keeping only the patches that have clear positive or negative label
    idx = (ys < 0.01) | (ys > 0.99)
    xs = xs[idx]
    ys = ys[idx]
    image_index = image_index[idx]

    print("Design matrix of size : ", xs.shape)
    print("Label matrix of size : ", ys.shape)
    print("Image index matrix of size : ", image_index.shape)
    print("DINOv3 Feature Extractor Script Complete")

    x_csv_path = args.output_csv + 'X_reduced.csv'
    y_csv_path = args.output_csv + 'y_reduced.csv'

    # Define the OutputDataFE Python object and load the X (features) and y (labels)
    output_data = OutputDataFE(xs, ys, x_csv_path, y_csv_path)

    # Call the Python class methods to output the X and y data from the DINOv3 feature extractor
    x_df = output_data.tensor_to_df_features()
    y_df = output_data.tensor_to_df_labels()
    print('Saved xs_reduced.csv and ys_reduced.csv to their output filepath directory')

    # Close your Weights and biases run
    wandb.finish()

def print_model_parameters(model, model_name="DINOv3"):
    """Print detailed information about model parameters"""
    
    print(f"\n{'='*60}")
    print(f"  {model_name} MODEL PARAMETERS")
    print(f"{'='*60}")
    
    total_params = 0
    trainable_params = 0
    
    print(f"\n{'Layer Name':<40} {'Shape':<25} {'Parameters':<15} {'Trainable'}")
    print("-" * 90)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
            trainable_status = "Yes"
        else:
            trainable_status = "No"
        
        shape_str = str(list(param.shape))
        print(f"{name:<40} {shape_str:<25} {param_count:<15,} {trainable_status}")
    
    print("-" * 90)
    print(f"{'TOTAL PARAMETERS':<40} {'':<25} {total_params:<15,}")
    print(f"{'TRAINABLE PARAMETERS':<40} {'':<25} {trainable_params:<15,}")
    print(f"{'NON-TRAINABLE PARAMETERS':<40} {'':<25} {total_params - trainable_params:<15,}")
    
    # Calculate model size in MB
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32 (4 bytes per param)
    print(f"{'ESTIMATED SIZE (MB)':<40} {'':<25} {model_size_mb:<15.2f}")
    print(f"{'='*60}")

def print_model_architecture(model, model_name="DINOv3"):
    """Print model architecture summary"""
    print(f"\n{'='*60}")
    print(f"  {model_name} ARCHITECTURE SUMMARY")
    print(f"{'='*60}")
    print(model)
    print(f"{'='*60}")
    
# Executes the main method from the main.py Python script
if __name__ == '__main__':
    # Calls the main function for the DINOv3 feature extractor (FE) script
    main()

