import os
from PIL import Image
import numpy as np

class LoadData():

  """
  Load custom image data from the Borebreen glacier in Svalbard Norway 
     to input ot the feature extractor oif DINOv3.
     ARGS: 
     image_dir (str): Input file path directory of the input images.
     labels_dir (str): Input file path directory of the input labels.
     
     RETURNS:
     images (list): Python list of all the input images for DINOv3's feature extractor.
     labels (list): Python list of all the input labels for DINOv3's feature extractor.  
  """

  def __init__(self, image_dir, labels_dir):
    self.image_dir = image_dir
    self.labels_dir = labels_dir

  def load_images(self):

    # List to store images
    images = []

    # Loop through all files in the directory
    for file_name in os.listdir(self.image_dir):
        file_path = os.path.join(self.image_dir, file_name)
    
        # Check if it's an image by extension
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            try:
                img = Image.open(file_path).convert("RGB")  # ensure RGB format
                images.append(img)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    print(f"Loaded {len(images)} images.")
    print(f"Image 1 Shpae {images[0].size}")
    print(f"Image 2 Shpae {images[1].size}")

    return images

  def load_labels(self):

    # List to store resized images
    labels = []

    # Loop through all files in the directory
    for file_name in os.listdir(self.labels_dir):
        file_path = os.path.join(self.labels_dir, file_name)
    
        # Check if it's an image by extension
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            try:
                img = Image.open(file_path)  # do NOT convert -> keep original mode
                img = img.resize((1024, 1024), Image.LANCZOS)  # resize to 1024x1024
                labels.append(img)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    print(f"Loaded {len(labels)} images resized to 1024x1024 (kept original mode).")
    print(f"Labels 1 Shpae {labels[0].size}")
    print(f"Labels 2 Shpae {labels[1].size}")

    return labels

  def print_image_info(self, labels):

    # Iterate and print min/max for each PIL image
    for idx, pil_img in enumerate(labels):
        arr = np.array(pil_img)  # convert PIL image to numpy array
        print(f"Image {idx+1}: min={arr.min()}, max={arr.max()}, mode={pil_img.mode}, size={pil_img.size}")

  def binary_labels_convert(self, labels):

    binary_labels = []

    for idx, pil_img in enumerate(labels):
        arr = np.array(pil_img)  # convert to numpy array

        min_val = arr.min()
        max_val = arr.max()

        # Map min -> 0, max -> 1
        bin_arr = np.where(arr == max_val, 255, 0).astype(np.uint8)

        print(f"Image {idx+1}: min={min_val}, max={max_val}, unique values after binarization={np.unique(bin_arr)}")

        binary_labels.append(Image.fromarray(bin_arr))

    print(f"Loaded {len(binary_labels)} images resized to 1024x1024 (kept original mode).")
    print(f"Labels 1 Shpae {binary_labels[0].size}")
    print(f"Labels 1 Shape {type(binary_labels[0])}")
    print(f"Labels 2 Type {binary_labels[1].size}")
    print(f"Labels 2 Type {type(binary_labels[1])}")

    return binary_labels

  def sequence_data_loading(self):

    images = self.load_images()

    labels = self.load_labels()

    self.print_image_info(labels)
  
    binary_labels = self.binary_labels_convert(labels)

    return images, binary_labels
