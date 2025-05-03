import numpy as np

TIMESTAMP = "0305011141"

features = np.load(f"ResNet_Features/train_features_{TIMESTAMP}.npy", allow_pickle=True)
labels = np.load(f"ResNet_Features/train_labels_{TIMESTAMP}.npy", allow_pickle=True)

print("Features shape:", features.shape)  
print("Labels shape:", labels.shape)

print("First feature vector:", features[0].shape)
print("First label:", labels[0])
