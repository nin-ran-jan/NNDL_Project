import numpy as np

features = np.load("ResNet_Features/train_features.npy", allow_pickle=True)
labels = np.load("ResNet_Features/train_labels.npy", allow_pickle=True)

print("Features shape:", features.shape)  
print("Labels shape:", labels.shape)

print("First feature vector:", features[0].shape)
print("First label:", labels[0])
