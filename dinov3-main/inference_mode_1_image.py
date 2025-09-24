import io
import os
import pickle
import tarfile
import urllib

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from load_data_1_image import LoadData

import wandb

import argparse

# Defines the ArgumentParser object
parser = argparse.ArgumentParser()

# Input parameters
parser.add_argument("--model_type", type=str, default='ViT7b')
parser.add_argument("--image_no", type=int, default=1)
parser.add_argument("--image_dir", type=str, default="/uoa/scratch/users/r02sw23/borebreen-drone-image-data-test/images/")
parser.add_argument("--label_dir", type=str, default="/uoa/scratch/users/r02sw23/borebreen-drone-image-data-test/masks/")
parser.add_argument("--output_path", type=str, default="/uoa/scratch/users/r02sw23/dinov3-main/saved_models/6/")
parser.add_argument("--filename", type=str, default="/uoa/scratch/users/r02sw23/dinov3-main/saved_models/finished_tests/5/fg_classifier.pkl")
# ------------------------------------------------------------------------

# Main funciton to sequence the Python script source code
def main():
    # Creates the ArgumentParser object in the main function
    args = parser.parse_args()

    test_image_dir = args.image_dir + f'borebreen_crop_drone_{args.image_no}.png'
    test_label_dir = args.label_dir + f'borebreen_crop_drone_{args.image_no}.png'

    load_data = LoadData(test_image_dir, test_label_dir)

    # Load the image data and their labels into the CUDA runtime
    test_image, test_label = load_data.sequence_data_loading()

    # Extract the first image and label from the list as a PIL image
    if isinstance(test_image, list):
        test_image = test_image[0]
    if isinstance(test_label, list):
        test_label = test_label[0]

    # Initialise the Weights and Biases run
    wandb.init(project=f"DINOv3 Segmentation Inference 1 Image {args.model_type} Test",
                name=f"Borebreen Image: {args.image_no}")

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
        weights='./pre_trained_weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth',
    )

    model.cuda()
    print(model)

    ##############################################################################################
    # Load trained Logistic Regression modelfrom a Pickle file
    with open(args.filename, "rb") as file:
        clf = pickle.load(file)
    
    ##############################################################################################
    # Parameters for running inferrence mode
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
    
    # Load test image to the Runtime
    #test_image_dir = args.image_dir + f'borebreen_crop_drone_{args.image_no}.png'
    #test_image = Image.open(test_image_dir).convert("RGB")
    test_image_resized = resize_transform(test_image)
    test_image_normalized = TF.normalize(test_image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    #test_label_dir = args.label_dir + f'borebreen_crop_drone_{args.image_no}.png'
    #label = load_test_labels(test_label_dir)

    # Etract the features from the test image
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            feats = model.get_intermediate_layers(test_image_normalized.unsqueeze(0).cuda(), n=range(n_layers), reshape=True, norm=True)
            x = feats[-1].squeeze().detach().cpu()
            dim = x.shape[0]
            x = x.view(dim, -1).permute(1, 0)
            
            #mask_i = test_label.split()[-1]
            #mask_i_resized = resize_transform(mask_i)
            mask_i_resized = resize_transform(test_label)
            mask_i_quantized = patch_quant_filter(mask_i_resized.unsqueeze(0)).squeeze().view(-1).detach().cpu()

    h_patches, w_patches = [int(d / PATCH_SIZE) for d in test_image_resized.shape[1:]]

    fg_score = clf.predict_proba(x)[:, 1].reshape(h_patches, w_patches)
    fg_score_mf = torch.from_numpy(signal.medfilt2d(fg_score, kernel_size=3))

    os.makedirs(args.output_path, exist_ok=True)

    plt.figure(figsize=(9, 3), dpi=300)
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(test_image_resized.permute(1, 2, 0))
    plt.title('input image')
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(fg_score)
    plt.title('foreground score')
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(fg_score_mf)
    plt.title('+ median filter')
    plt.savefig(args.output_path + f'model_outputs_plot_borebreen{args.image_no}.png')
    plt.close()

    y_pred = clf.predict(x).reshape(h_patches, w_patches)
    
    # Convert tensors to numpy arrays and flatten for metric calculation
    # Convert mask_i_quantized to numpy if it's a tensor
    if isinstance(mask_i_quantized, torch.Tensor):
        mask_i_flat = mask_i_quantized.long().numpy().flatten()
    else:
        mask_i_flat = mask_i_quantized.long().flatten()
    
    # Convert y_pred to numpy and flatten
    if isinstance(y_pred, torch.Tensor):
        y_pred_flat = y_pred.numpy().flatten()
    else:
        y_pred_flat = y_pred.flatten()
    
    # Debug: Print shapes and data types
    print(f"mask_i_flat shape: {mask_i_flat.shape}, dtype: {mask_i_flat.dtype}")
    print(f"y_pred_flat shape: {y_pred_flat.shape}, dtype: {y_pred_flat.dtype}")
    print(f"mask_i_flat unique values: {np.unique(mask_i_flat)}")
    print(f"y_pred_flat unique values: {np.unique(y_pred_flat)}")

    # Stores the results from the test dataset to the screen
    dsc = metrics.f1_score(mask_i_flat, y_pred_flat, zero_division=1)
    iou = metrics.jaccard_score(mask_i_flat, y_pred_flat)
    acc = metrics.accuracy_score(mask_i_flat, y_pred_flat)
    pre = metrics.precision_score(mask_i_flat, y_pred_flat, zero_division=1)
    rec = metrics.recall_score(mask_i_flat, y_pred_flat, zero_division=1)

    # Check the accuracy on the test dataset. If this is too low compared to train it indicates overfitting on the training data
    print("Dice Score Coefficiant = ", dsc)
    print("IoU = ", iou)
    print("Accuracy = ", acc)
    print("Precision = ", pre)
    print("Recall = ", rec)

    plt.figure(figsize=(9, 3), dpi=300)
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(test_image_resized.permute(1, 2, 0))
    plt.title('input image')
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(y_pred)
    plt.title('foreground score')
    plt.savefig(args.output_path + f'model_predictions_plot_borebreen{args.image_no}.png')
    plt.close()

    # Close your Weights and biases run
    wandb.finish()

# Executes the main method from the main.py Python script
if __name__ == '__main__':
    # Calls the main function for the DINOv3 feature extractor (FE) script
    main()