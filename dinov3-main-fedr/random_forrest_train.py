import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import os
import pickle

import wandb

# Hyperparamters and filepath directories
est = 100
#x_file_path = '/uoa/scratch/users/r02sw23/dinov3-main-fedr/output_csv_images/X_reduced.csv'
#y_file_path = '/uoa/scratch/users/r02sw23/dinov3-main-fedr/output_csv_images/y_reduced.csv'
x_file_path = '/uoa/scratch/users/r02sw23/dinov3-main-fedr/output_csv_images/X.csv'
y_file_path = '/uoa/scratch/users/r02sw23/dinov3-main-fedr/output_csv_images/y.csv'
save_root = '/uoa/scratch/users/r02sw23/dinov3-main-fedr/RF_models/'
model_type = 'RF'
# -----------------------------------------

# Initialise the Weights and Biases run
wandb.init(project=f"DINOv3 Segmentation {model_type}",
            name=f"Test Run 2")

# Copy your config
config = wandb.config

data = pd.read_csv(x_file_path)
data = data.astype('float32')
#X = data.iloc[:65536,:]
X = data
print(f"Dataset features loaded with shape: {X.shape}")

data = pd.read_csv(y_file_path)
data = data.astype('float32')
#y = data.iloc[:65536,:]
y = data
print(f"Dataset labels loaded with shape: {y.shape}")

y = y.values.ravel()
print(f"Dataset labels loaded with shape: {y.shape}")

print("Y Unique Values: ", np.unique(y))
y = np.round(y)
print("Y Unique Values: ", np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20,
                                                    random_state=1)

clf = RandomForestClassifier(n_estimators=est, n_jobs=-1)

print('Random Forrest Classifier Training')
clf.fit(X_train, y_train)
print('Random Forrest Classifier Training Complete')

print('Random Forrest Classifier Inference Mode')
y_pred = clf.predict(X_test)
print('Random Forrest Classifier Inference Mode Complete')

# Stores the results from the test dataset and prints them to the screen
dsc = metrics.f1_score(y_test, y_pred, zero_division=1)
iou = metrics.jaccard_score(y_test, y_pred)
acc = metrics.accuracy_score(y_test, y_pred)
pre = metrics.precision_score(y_test, y_pred, zero_division=1)
rec = metrics.recall_score(y_test, y_pred, zero_division=1)

print("Dice Score Coefficient = ", dsc)
print("IoU = ", iou)
print("Accuracy = ", acc)
print("Precision = ", pre)
print("Recall = ", rec)

os.makedirs(save_root, exist_ok=True)

model_path = os.path.join(save_root, f"fg_classifier{est}_est.pkl")
with open(model_path, "wb") as f:
    pickle.dump(clf, f)

# Close your Weights and biases run
wandb.finish()





