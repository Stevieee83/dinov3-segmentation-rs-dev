# dinov3-main

The DINOv3 Main part of this repository was used to run the DINOv3 examples out of the box from the DINOv3 GitHub repository from Meta AI FAIR.

dinov37B_borebreen.py - Python script used to load training images from the Borebreen UaV dataset to the DINOv3 ViT-7B model that has be pre-trained on natural images and perform feature extraction for foreground segmentation. 
The extracted features are then used to train and tune the regularisation C parameter of a Logistic regression model and test it on a single image from the Borebreen UaV test dataset. The test image configured to load to
the script is borebreen_crop_drone_3.png. The trained weights from the logistic regression model is output to a Pickle file fg_classifier.pkl at the end of the script.

dinov37B_borebreenRS.py - Python script used to load training images from the Borebreen UaV dataset to the DINOv3 ViT-7B model that has be pre-trained on remote sensing and airial images and perform feature extraction for 
foreground segmentation. The extracted features are then used to train and tune the regularisation C parameter of a Logistic regression model and test it on a single image from the Borebreen UaV test dataset. The test 
image configured to load to the script is borebreen_crop_drone_3.png. The trained weights from the logistic regression model is output to a Pickle file fg_classifier.pkl at the end of the script.

dinov3FE.py - Extracts features with the DINOv3 ViT-L trianed on natural images and outputs the shapes of the full size and reduced size NumPy arrays to the screen.

dinov3_7b.py - Example Jupyter Notebook from the Meta AI (FAIR) DINOv3 GitHub repository converted to a Pyhton script to run on a Slurm HPC with the ViT-7B model.

dinov3_7b_infer.py - Example Jupyter Notebook from the Meta AI (FAIR) DINOv3 GitHub repository converted to a Pyhton script to run on a Slurm HPC with the ViT-7B model after feature extraction. The script loads no images data 
in it's present state. The code for Leave One out Cross Validation and Logistic Regression inference is present.

dinov3segprint.py - Example Jupyter Notebook from the Meta AI (FAIR) DINOv3 GitHub repository converted to a Pyhton script to run on a Slurm HPC with the ViT-7B model. The images that are loaded to the script for feature extraction
, Leave One Out cross validation and testing on a trained Logistic Regression are all downloaded from the Meta example images server. The shapes of all the lists, NumPy arrays and model outputs etc are printed out the screen. The
script was used to learn how the example notebook worked for DINOv3 from Meta AI (FAIR) in preperation for loading custom data to the models.

hubconf.py - Meta AI (FAIR) script required for running DINOv3 model locally on a PC or HPC system as opposed to loading in the models from the PyTorch torch model hub API.

inference_mode_1_image.py - Performs feature extraction on a single custom image from the UaV Borebreen dataset, loads a saved Pickle Logistic Regression model file and runs inference mode on the custom image where features were 
extracted from DINOv3 at the start of the script. The segmentation metric results were outuput to the screen.

inference_mode_images.py - Performs feature extraction on custom images from the UaV Borebreen dataset, loads a saved Pickle Logistic Regression model file and runs inference mode on the custom images where features were 
extracted from DINOv3 at the start of the script. The segmentation metric results were outuput to the screen.

load_data.py -  Loads image data and labels to the srcript in preperation for DINOv3.

load_data_1_image.py -  Loads 1 image and the respective label to the srcript in preperation for DINOv3.

requirements-dev.txt - requirements from the DINOv3 GitHub repository for segmentation.

requirements.txt - requirements from the DINOv3 GitHub repository.

run_infer.sh - Shell script for the HPC system to run inference mode tests with DINOv3.

run_slurm.sh - Shell script for the HPC system to run tests with DINOv3.

run_slurm_fe.sh - Shell script for the HPC system to run feature extraction mode tests with DINOv3.

run_slurm_seg_pr.sh - Shell script for the HPC system to run tests with DINOv3 and print out list/array and model shape information out to the screen.
