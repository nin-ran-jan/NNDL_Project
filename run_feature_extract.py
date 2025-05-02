import pandas as pd
import numpy as np
from models.ResNet_feature_extract import extract_features_and_labels

if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    df["id"] = df["id"].apply(lambda x: str(x).zfill(5))
    df_subset = df.head(10)

    TRAIN_VIDEO_DIR = "data/train"
    sequences, labels = extract_features_and_labels(TRAIN_VIDEO_DIR, df_subset)

    print("Extracted sequences shape:", sequences.shape)
    print("Extracted labels shape:", labels.shape)

    np.save("ResNet_Features/train_features.npy", sequences)
    np.save("ResNet_Features/train_labels.npy", labels)
    print("Saved to ResNet_Features/train_features.npy and ResNet_Features/train_labels.npy")