import torch
import numpy as np
import os
import csv
import time # For unique timestamps for submission file if needed

# Import your models and dataset
# User's script has: from models.Vit_Transformer import ViTTransformer
from models.Vit_Transformer import ViTTransformer 
from datasets.feature_dataset import AccidentFeatureDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence # For collate_fn

# --- Configuration Section ---
# Timestamps and Paths
# Timestamp for this specific test run (used for naming the submission file)
CURRENT_TEST_RUN_TIMESTAMP = time.strftime("%Y%m%d%H%M%S") 

# !!! IMPORTANT: Update these to match your TRAINED ViTTransformer model and TEST ViT features !!!
# Timestamp/ID of the TRAINED ViTTransformer model checkpoint you want to use for testing
MODEL_CHECKPOINT_TIMESTAMP = "20250509170638" # Example: The TRAIN_RUN_TIMESTAMP of the saved best model

# Timestamp/ID of the TEST ViT feature set you are using
# This should help locate the correct test feature files.
# Assuming a similar structure to your training data, e.g., features are in a 'run_DATA_TIMESTAMP' subdir
TEST_DATA_TIMESTAMP = "250509_173651" 

# Feature directory structure (similar to your training script)
FEATURE_DIR_BASE = "CLIP_ViT_Features_Test_clip-vit-large-patch14"  # Base dir for ViT features
TEST_FEATURE_SUBDIR = f"run_{TEST_DATA_TIMESTAMP}" # Subdirectory for this specific TEST feature set
                                                   # OR, if your test features are structured differently, adjust accordingly.
                                                   # For example, if it's just "test_batch_..." directly under FEATURE_DIR_BASE,
                                                   # then full_feature_dir would be just FEATURE_DIR_BASE.

# Path to the trained model checkpoint
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FILENAME = f"ViTTransformer_best_{MODEL_CHECKPOINT_TIMESTAMP}.pth"
MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)

# Model Hyperparameters for ViTTransformer (MUST match the loaded checkpoint's model)
VIT_FEATURE_DIM = 768      
MODEL_DIM = 512            
N_HEADS = 8                
NUM_ENCODER_LAYERS = 4     
DIM_FEEDFORWARD = 1024     
# Dropout in __init__ is an architectural choice. model.eval() handles its behavior during inference.
# Set this to the dropout value the loaded model was trained with.
MODEL_DROPOUT = 0.25 # Example, ensure this matches the loaded model's config

# Test Hyperparameters
BATCH_SIZE = 32
# --- End Configuration Section ---

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Current Test Run Timestamp: {CURRENT_TEST_RUN_TIMESTAMP}")
print(f"Loading model checkpoint: {MODEL_CHECKPOINT_PATH}")


def load_test_features_and_ids(feature_dir_path):
    """
    Loads all test features and their IDs from .npy files in the specified directory.
    Adjusted for a naming pattern like:
    test_features_saving_batch_<INDEX>.npy
    test_ids_saving_batch_<INDEX>.npy 
    (This is an assumption based on your training script's 'saving_batch' pattern. 
     If your test files are named differently, e.g., using TEST_DATA_TIMESTAMP, adjust this function.)
    """
    all_features = []
    all_ids = []
    
    if not os.path.isdir(feature_dir_path):
        print(f"Error: Test feature directory not found: {feature_dir_path}")
        return None, None

    try:
        # Adapt this logic if your test file naming is different
        # E.g. if they include TEST_DATA_TIMESTAMP like: test_features_batch0_TESTDATATIMESTAMP.npy
        # This example assumes a pattern like 'test_features_saving_batch_X.npy'
        feature_batch_indices = sorted(list(set(
            f.split("_")[4].split(".")[0] 
            for f in os.listdir(feature_dir_path)
            if f.startswith("test_features_saving_batch") and f.endswith(".npy") 
        )))

        if not feature_batch_indices:
            print(f"Error: No test feature batch files found in {feature_dir_path} matching 'test_features_saving_batch_X.npy'.")
            return None, None
        print(f"Found test feature batch indices: {feature_batch_indices}")

    except Exception as e:
        print(f"Error discovering test feature batches: {e}")
        return None, None

    for batch_idx in feature_batch_indices:
        # Adjust file naming to match your test set
        feature_file = f"test_features_saving_batch_{batch_idx}.npy"
        id_file = f"test_ids_saving_batch_{batch_idx}.npy" # Assuming a similar naming for ID files
        
        feature_path = os.path.join(feature_dir_path, feature_file)
        ids_path = os.path.join(feature_dir_path, id_file)

        if not os.path.exists(feature_path):
            print(f"Warning: Test feature file not found: {feature_path}. Skipping.")
            continue
        if not os.path.exists(ids_path):
            print(f"Warning: Test ID file not found: {ids_path}. Skipping.")
            # Decide if you want to skip the feature file too, or handle missing IDs differently
            continue 
            
        try:
            features = np.load(feature_path, allow_pickle=True)
            ids = np.load(ids_path, allow_pickle=True)
            all_features.extend(list(features)) 
            all_ids.extend(list(ids))     
        except Exception as e:
            print(f"Error loading test batch {batch_idx}: {e}")
            continue
            
    if not all_features or not all_ids:
        print("No test data loaded. Please check feature directory and file naming.")
        return None, None
    
    if len(all_features) != len(all_ids):
        print(f"Warning: Mismatch in number of loaded features ({len(all_features)}) and IDs ({len(all_ids)}).")
        # Decide on error handling, e.g., take the minimum length or exit.
        # For now, we'll proceed but this indicates an issue.
        
    print(f"Loaded {len(all_features)} total test sequences and {len(all_ids)} IDs.")
    return all_features, all_ids


def collate_sequences_test(batch):
    """
    Collate function for test data. Pads sequences and creates padding masks.
    The 'batch' here will be a list of (sequence_features, dummy_frame_labels, dummy_binary_label)
    We only care about sequence_features for padding.
    """
    sequences = [item[0] for item in batch] # item[0] is sequence_features

    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    lengths = torch.tensor([len(seq) for seq in sequences])
    max_len = sequences_padded.size(1)
    padding_mask = torch.arange(max_len)[None, :] >= lengths[:, None] 

    # We don't need to return labels for the test script's model input
    return sequences_padded, padding_mask


# --- Data Loading for Test Set ---
# Construct the full path to the directory containing batched TEST feature files
# This needs to align with how your test features are stored.
# Example: "processed_data/CLIP_ViT_Features_clip-vit-large-patch14/run_test_YYYYMMDDHHMMSS_ViT_Test_Features"
full_test_feature_dir = os.path.join(FEATURE_DIR_BASE, TEST_FEATURE_SUBDIR) 
print(f"Attempting to load test data from: {full_test_feature_dir}")

# Load all test features and corresponding IDs
# The load_test_features_and_ids function needs to be robust to your file naming.
test_features_all, test_ids_all = load_test_features_and_ids(full_test_feature_dir)

if test_features_all is None or test_ids_all is None:
    print("Exiting due to test data loading issues.")
    exit()

# Create dummy frame labels for the AccidentFeatureDataset
# The actual content of these labels doesn't matter for inference if only seq_probs are used.
if not test_features_all: # Handle case where no features were loaded
    print("No test features loaded. Exiting.")
    exit()
    
# Ensure test_features_all[0] is a numpy array and has a shape attribute
# This assumes all sequences in test_features_all are numpy arrays.
# If they are lists of lists, convert them to numpy arrays first or adjust seq_len_example logic.
try:
    if isinstance(test_features_all[0], np.ndarray):
        seq_len_example = test_features_all[0].shape[0]
    elif isinstance(test_features_all[0], torch.Tensor): # If already tensors
        seq_len_example = test_features_all[0].size(0)
    else: # Fallback or error if type is unexpected
        print(f"Warning: Unexpected feature type: {type(test_features_all[0])}. Assuming length 100 for dummy labels.")
        seq_len_example = 100 # Or handle as an error
except (IndexError, AttributeError) as e:
    print(f"Error determining sequence length for dummy labels: {e}. Exiting.")
    exit()
    
dummy_frame_labels = np.zeros((len(test_features_all), seq_len_example, 1), dtype=np.float32)

# Create dataset and dataloader for the test set
test_dataset = AccidentFeatureDataset(test_features_all, dummy_frame_labels) # Pass all loaded features
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_sequences_test)


# --- Model Instantiation and Loading ---
model = ViTTransformer(
    feature_dim=VIT_FEATURE_DIM,
    model_dim=MODEL_DIM,
    nhead=N_HEADS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=MODEL_DROPOUT # This should match the dropout the model was trained with
).to(device)

try:
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
    print(f"Successfully loaded model checkpoint from: {MODEL_CHECKPOINT_PATH}")
except FileNotFoundError:
    print(f"Error: Model checkpoint not found at {MODEL_CHECKPOINT_PATH}")
    print("Please ensure MODEL_CHECKPOINT_TIMESTAMP and CHECKPOINT_DIR are correct.")
    exit()
except Exception as e:
    print(f"Error loading model checkpoint: {e}")
    exit()

model.eval() # Set the model to evaluation mode

# --- Inference ---
all_scores = []
# Note: test_ids_all are already loaded and correspond to test_features_all

with torch.no_grad():
    for batch_idx, (sequences_padded, padding_mask) in enumerate(test_loader):
        print(f"Processing test batch {batch_idx + 1}/{len(test_loader)}...")
        sequences_padded = sequences_padded.to(device)
        padding_mask = padding_mask.to(device)
        
        # Get model predictions (frame_probs, seq_probs)
        # We need seq_probs for the submission.
        _, seq_probs = model(sequences_padded, src_key_padding_mask=padding_mask) 
        
        scores_batch = seq_probs.cpu().numpy()
        if scores_batch.ndim > 1 and scores_batch.shape[1] == 1: # Handle shape (N, 1)
            scores_batch = scores_batch.squeeze(1)
        all_scores.extend(scores_batch.tolist())

# Ensure the number of scores matches the number of IDs
if len(all_scores) != len(test_ids_all):
    print(f"Critical Error: Number of scores ({len(all_scores)}) does not match number of IDs ({len(test_ids_all)}).")
    print("This might be due to issues in data loading or processing. Submission will be incorrect.")
    # You might want to exit here or handle this error more robustly
else:
    print(f"Generated {len(all_scores)} scores for {len(test_ids_all)} test videos.")

# --- Submission File ---
submission_dir = "submissions"
os.makedirs(submission_dir, exist_ok=True)
# Use the CURRENT_TEST_RUN_TIMESTAMP for a unique submission file name
submission_filename = f"submission_ViTTransformer_{CURRENT_TEST_RUN_TIMESTAMP}.csv"
submission_path = os.path.join(submission_dir, submission_filename)

with open(submission_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'score'])  # Header row
    for vid_id, score in zip(test_ids_all, all_scores): # Use the globally loaded test_ids_all
        writer.writerow([vid_id, f"{score:.6f}"]) # Format score 

print(f"Successfully saved predictions to {submission_path}")
