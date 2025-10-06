import os
import pickle

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression

import torch
import torchvision
import torchvision.transforms.functional as TF

from tqdm import tqdm

from load_data import LoadData
from output_data_fe import OutputDataFE

import argparse

# Defines the ArgumentParser object
parser = argparse.ArgumentParser()

# Checks the version of PyTorch and torchvision
print("PyTorch version: ", torch.__version__)
print("torchvision version: ", torchvision.__version__)

parser.add_argument("--model_type", type=str, default='ViTL493M-Natural')
parser.add_argument("--max_block_no", type=int, default=23)
parser.add_argument("--image_size", type=int, default=512)

parser.add_argument("--run", type=int, default=1)
parser.add_argument("--file_path_csv", type=str, default='./output_csv_data/')
parser.add_argument("--file_path_images", type=str, default='images/')
parser.add_argument("--file_path_labels", type=str, default='labels/')
parser.add_argument("--image_dir", type=str, default="C:/Users/r02sw23/PycharmProjects/pythonProject2/dinov3-dora-classifier-segmentation-tests/train/images")
parser.add_argument("--labels_dir", type=str, default="C:/Users/r02sw23/PycharmProjects/pythonProject2/dinov3-dora-classifier-segmentation-tests/train/masks")
parser.add_argument("--test_image_path", type=str, default="C:/Users/r02sw23/PycharmProjects/pythonProject2/dinov3-dora-classifier-segmentation-tests/test/images/borebreen_crop_drone_1_r0c0_512.png")
parser.add_argument("--model_path", type=str,
                    default='C:/model_weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
parser.add_argument("--output_path", type=str,
                    default="C:/Users/r02sw23/PycharmProjects/pythonProject2/dinov3-foreground-segmentation-tests-natural/test_results/")
parser.add_argument("--DINOV3_GITHUB_LOCATION", type=str, default="facebookresearch/dinov3")

# Load the DINOv3 model backbone and send to the CUDA device
# examples of available DINOv3 models:
# parser.add_argument("--MODEL_NAME", type=str, default="dinov3_vits16")
# parser.add_argument("--MODEL_NAME", type=str, default="dinov3_vits16plus")
# parser.add_argument("--MODEL_NAME", type=str, default="dinov3_vitb16")
parser.add_argument("--MODEL_NAME", type=str, default="dinov3_vitl16")
# parser.add_argument("--MODEL_NAME", type=str, default="dinov3_vith16plus")
# parser.add_argument("--MODEL_NAME", type=str, default="dinov3_vit7b16")
# ------------------------------------------------------------------------


def main():
    # Creates the ArgumentParser object in the main function
    args = parser.parse_args()

    load_data = LoadData(args.image_dir, args.labels_dir, args.image_size)

    os.makedirs(args.file_path_csv + args.file_path_images, exist_ok=True)
    os.makedirs(args.file_path_csv + args.file_path_labels, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    # Sets the device and the random seeds
    if torch.cuda.is_available():
        # Sets the deice to CUDA GPU
        device = 'cuda'
        # Set random seed for GPU
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    else:
        # Sets the deice to CPU
        device = 'cpu'
        # Set random seed for CPU
        torch.cuda.manual_seed(42)

    print("Device: ", device)

    ####################################################################################################################################################
    if os.getenv("DINOV3_LOCATION") is not None:
        DINOV3_LOCATION = os.getenv("DINOV3_LOCATION")
    else:
        DINOV3_LOCATION = args.DINOV3_GITHUB_LOCATION
        print(f"DINOv3 location set to {DINOV3_LOCATION}")
        
    model = torch.hub.load(repo_or_dir=DINOV3_LOCATION,
                                     model=args.MODEL_NAME,
                                     source="local" if DINOV3_LOCATION != args.DINOV3_GITHUB_LOCATION else "github",
                                     weights=args.model_path,
    )

    # Push the DINOv3 model to the GPU and print out the architecture to the screen
    model.cuda()
    print('DINOv3 Pre-Trained Model Architecture')
    print(type(model))
    print(model)

    print('Model LoRA Output')
    image = torch.rand(1, 3, 512, 512).cuda()
    y = model(image)
    print(y.shape)
    print(y)
    ####################################################################################################################################################

    # Load the image data and their labels into the CUDA runtime
    images, labels = load_data.sequence_data_loading()

    n_images = len(images)
    assert n_images == len(labels), f"{len(images)=}, {len(labels)=}"

    ##############################################################################################
    # Building the Per-Patch Label Map
    PATCH_SIZE = 16
    IMAGE_SIZE = args.image_size

    # quantization filter for the given patch size
    patch_quant_filter = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
    patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))

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
    xs = []
    ys = []
    image_index = []

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    # MODEL_TO_NUM_LAYERS = {
    #     MODEL_DINOV3_VITS: 12,
    #     MODEL_DINOV3_VITSP: 12,
    #     MODEL_DINOV3_VITB: 12,
    #     MODEL_DINOV3_VITL: 24,
    #     MODEL_DINOV3_VITHP: 32,
    #     MODEL_DINOV3_VIT7B: 40,
    # }

    n_layers = args.max_block_no + 1

    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i in tqdm(range(n_images), desc="Processing images"):
                # Loading the ground truth
                mask_i = labels[i]
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
                xs.append(feats[-1].squeeze().view(dim, -1).permute(1, 0).detach().cpu())

                image_index.append(i * torch.ones(ys[-1].shape))

    # Concatenate all lists into torch tensors
    xs = torch.cat(xs)
    ys = torch.cat(ys)
    ys_round = torch.round(ys)
    image_index = torch.cat(image_index)

    x_path = args.file_path_csv + args.file_path_images + '/x_full.csv'
    y_path = args.file_path_csv + args.file_path_labels + '/y_full.csv'

    # Define the OutputDataFE Python object and load the X (features) and y (labels)
    output_data = OutputDataFE(xs, ys_round, x_path, y_path)

    # Call the Python class methods to output the X and y data from the DINOv3 feature extractor
    output_data.tensor_to_df_features()
    output_data.tensor_to_df_labels()

    print("Design matrix of size full : ", xs.shape)
    print("Label matrix of size full : ", ys.shape)
    print("DINOv3 Feature Extractor Script Complete")

    # keeping only the patches that have clear positive or negative label
    idx = (ys < 0.01) | (ys > 0.99)
    xs = xs[idx]
    ys = ys[idx]
    image_index = image_index[idx]

    x_path = args.file_path_csv + args.file_path_images + '/x.csv'
    y_path = args.file_path_csv + args.file_path_labels + '/y.csv'

    # Define the OutputDataFE Python object and load the X (features) and y (labels)
    output_data = OutputDataFE(xs, ys, x_path, y_path)

    # Call the Python class methods to output the X and y data from the DINOv3 feature extractor
    output_data.tensor_to_df_features()
    output_data.tensor_to_df_labels()

    print("Design matrix of size : ", xs.shape)
    print("Label matrix of size : ", ys.shape)
    print("DINOv3 Feature Extractor Script Complete")

    ##############################################################################################
    # Training a classifier and model selection
    cs = np.logspace(-7, 0, 8)
    scores = np.zeros((n_images, len(cs)))

    print('Leave One Out Cross Validation Underway')
    for i in range(n_images):
        # We use leave-one-out so train will be all but image i, val will be image i
        print('validation using image_{:02d}.jpg'.format(i + 1))
        train_selection = image_index != float(i)
        fold_x = xs[train_selection].numpy()
        fold_y = (ys[train_selection] > 0).long().numpy()
        val_x = xs[~train_selection].numpy()
        val_y = (ys[~train_selection] > 0).long().numpy()

        plt.figure()
        for j, c in enumerate(cs):
            print("training logisitic regression with C={:.2e}".format(c))
            clf = LogisticRegression(random_state=0, C=c, max_iter=10000).fit(fold_x, fold_y)
            output = clf.predict_proba(val_x)
            precision, recall, thresholds = precision_recall_curve(val_y, output[:, 1])
            s = average_precision_score(val_y, output[:, 1])
            scores[i, j] = s
            plt.plot(recall, precision, label='C={:.1e} AP={:.1f}'.format(c, s * 100))

        plt.grid()
        plt.xlabel('recall')
        plt.title('image_{:02d}.jpg'.format(i + 1))
        plt.ylabel('precision')
        plt.axis([0, 1, 0, 1])
        plt.legend()
        plt.savefig(args.output_path + f'cross_validation_plot_image_{i+1}.png')
        #plt.show()
        plt.close()

    ##############################################################################################
    # Choosing the best C
    plt.figure(figsize=(3, 2), dpi=300)
    plt.rcParams.update({
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "axes.labelsize": 5,
    })
    plt.plot(scores.mean(axis=0))
    plt.xticks(np.arange(len(cs)), ["{:.0e}".format(c) for c in cs])
    plt.xlabel('data fit C')
    plt.ylabel('average AP')
    plt.grid()
    plt.savefig(args.output_path + f'overall_cv_plot_best_c.png')
    #plt.show()
    plt.close()

    ##############################################################################################
    # Retraining with the optimal regularisation of C = 0.1
    clf = LogisticRegression(random_state=0, C=1e-1, max_iter=100000, verbose=2).fit(xs.numpy(), (ys > 0).long().numpy())

    ##############################################################################################
    # Test images and inferrence
    test_image = Image.open(args.test_image_path).convert("RGB")
    test_image_resized = resize_transform(test_image)
    test_image_normalized = TF.normalize(test_image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            feats = model.get_intermediate_layers(test_image_normalized.unsqueeze(0).cuda(), n=range(n_layers),
                                                  reshape=True, norm=True)
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
    plt.savefig(args.output_path + f'test_image_subplot.png')
    #plt.show()
    plt.close()

    ##############################################################################################
    # Saving the model for future use as a pickle file
    save_root = '.'
    output_model_path = os.path.join(save_root, "fg_classifier.pkl")
    with open(output_model_path, "wb") as f:
        pickle.dump(clf, f)


# Executes the main method from the main.py Python script
if __name__ == '__main__':
    # Calls the main function for the DINOv3 feature extractor (FE) script
    main()
