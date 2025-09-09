import os
from PIL import Image
import numpy as np

class LoadData():

  """
  Load custom image data from the Borebreen glacier in Svalbard Norway 
     to input to the feature extractor of DINOv3.
     ARGS: 
     image_dir (str): Input file path directory of the input image.
     labels_dir (str): Input file path directory of the input label.
     
     RETURNS:
     images (list): Python list of all the input images for DINOv3's feature extractor.
     labels (list): Python list of all the input labels for DINOv3's feature extractor.  
  """

  def __init__(self, image_dir, labels_dir):
    self.image_dir = image_dir
    self.labels_dir = labels_dir

  def load_images(self):
    
    # Check if it's an image by extension
    image = Image.open(self.image_dir).convert("RGB")  # ensure RGB format
    print(f"Loaded Image Shape {image.size}")

    return image

  def load_labels(self):
                
    label = Image.open(self.labels_dir).convert('L')
    print(f"Loaded Label Shape {label.size}")
    
    return label

  def print_image_info(self, label):

    min_val, max_val = label.getextrema()
    print(f"Image: min={min_val}, max={max_val}, mode={label.mode}, size={label.size}")

  def binary_labels_convert(self, label, new_min=0, new_max=255):

    if label.mode != "L":
      raise ValueError("Image must be in grayscale ('L') mode.")
        
    # Get current min/max
    extrema = label.getextrema()
    old_min, old_max = extrema

    print("Label old mininmum pixel value:", old_min)
    print("Label old maximum pixel value:", old_max)

    if old_min == old_max:
      # Avoid divide-by-zero; image is flat
      return label.point(lambda p: new_min)

    # Linear rescale
    scale = (new_max - new_min) / (old_max - old_min)
    offset = new_min - old_min * scale

    new_label = label.point(lambda p: int(p * scale + offset))

    binary_label.append(new_label)

    new_min, new_max = new_label.getextrema()
    print("Label new mininmum pixel value:", new_min)
    print("Label new maximum pixel value:", new_max)

    return binary_label

  def sequence_data_loading(self):

    images = self.load_images()

    labels = self.load_labels()

    self.print_image_info(labels)
  
    binary_labels = self.binary_labels_convert(labels)

    return images, binary_labels
