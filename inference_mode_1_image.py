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

from load_data import LoadData

import wandb

# Input parameters
model_type = 'ViT7b'                                                                                                # DINOv3 ViT image encoder type
test_image_path = "/uoa/scratch/users/r02sw23/borebreen-drone-image-data-test/images/borebreen_crop_drone_1.png"    # Test images and inferrence
test_labels_path = "/uoa/scratch/users/r02sw23/borebreen-drone-image-data-test/masks/borebreen_crop_drone_1.png"    # Test images and inferrence
output_path = "/uoa/scratch/users/r02sw23/dinov3-main/saved_models/5/"
image_no = 1

# Initialise the Weights and Biases run
wandb.init(project=f"DINOv3 Segmentation FE {model_type} Test",
            name=f"Borebreen Image: {image_no}")

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
with open(filename, "rb") as file:
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

# image resize transform to dimensions divisible by patch size
def resize_transform(
    mask_image: Image,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))

##############################################################################################
# Load test image to the Runtime
test_image = Image.open(test_image_path).convert("RGB")
test_image_resized = resize_transform(test_image)
test_image_normalized = TF.normalize(test_image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)

# Etract the features from the test image
with torch.inference_mode():
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        feats = model.get_intermediate_layers(test_image_normalized.unsqueeze(0).cuda(), n=range(n_layers), reshape=True, norm=True)
        x = feats[-1].squeeze().detach().cpu()
        dim = x.shape[0]
        x = x.view(dim, -1).permute(1, 0)

h_patches, w_patches = [int(d / PATCH_SIZE) for d in test_image_resized.shape[1:]]

fg_score = clf.predict_proba(x)[:, 1].reshape(h_patches, w_patches)
fg_score_mf = torch.from_numpy(signal.medfilt2d(fg_score, kernel_size=3))

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
plt.savefig(output_path + f'model_outputs_plot_borebreen{image_no}.png')
plt.close()

def load_test_labels(test_labels_path):
    
    img = Image.open(test_labels_path).convert("L")
    
    # Convert to numpy
    arr = np.array(img)

    # Scale so that the maximum becomes 1 (others scaled accordingly)
    arr = (arr / arr.max()) * 1

    # Convert back to uint8 (values 0 or 1)
    arr_uint8 = arr.astype(np.uint8)

    print('Array shape: ',arr_uint8.shape)
    print('Array Max Value: ',arr_uint8.max())  # should be 1
    print('Array Min Value: ',arr_uint8.min())  # should be 0
    print('Unique values: ', np.unique(arr_uint8))
    print('Array type: ',arr_uint8.dtype)       # uint8

y_pred = clf.predict(x)[:, 1].reshape(h_patches, w_patches)
y_test = load_test_labels()

# Stores the results from the test dataset to the screen
dsc = metrics.f1_score(y_test, y_pred, zero_division=1)
iou = metrics.jaccard_score(y_test, y_pred)
acc = metrics.accuracy_score(y_test, y_pred)
pre = metrics.precision_score(y_test, y_pred, zero_division=1)
rec = metrics.recall_score(y_test, y_pred, zero_division=1)

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
plt.savefig(output_path + f'model_outputs_plot_borebreen{image_no}.png')
plt.close()

# Close your Weights and biases run
wandb.finish()
