import numpy as np

TIMESTAMP = "0305050051"
MODE = "test"

features = np.load(f"ResNet_Features/{MODE}_features_{TIMESTAMP}.npy", allow_pickle=True)
if MODE == "train":
    labels = np.load(f"ResNet_Features/{MODE}_labels_{TIMESTAMP}.npy", allow_pickle=True)

print("Features shape:", features.shape) 
if MODE == "train": 
    print("Labels shape:", labels.shape)

print("First feature vector:", features[0].shape)
if MODE == "train":
    print("First label:", labels[0])
