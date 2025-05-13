import torch
import numpy as np
import os
import csv
import time

from models.Vit_Transformer import ViTTransformer 
from datasets.feature_dataset import AccidentFeatureDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

CURRENT_TEST_RUN_TIMESTAMP = time.strftime("%Y%m%d%H%M%S") 

MODEL_CHECKPOINT_TIMESTAMP = "20250511163435"

TEST_DATA_TIMESTAMP = "250509_185359" 

FEATURE_DIR_BASE = "CLIP_ViT_Features_Test_clip-vit-large-patch14" 
TEST_FEATURE_SUBDIR = f"run_{TEST_DATA_TIMESTAMP}" 

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FILENAME = f"ViTTransformer_best_{MODEL_CHECKPOINT_TIMESTAMP}.pth"
MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)

VIT_FEATURE_DIM = 768      
MODEL_DIM =  128           
N_HEADS =  4               
NUM_ENCODER_LAYERS = 2     
DIM_FEEDFORWARD = 640    
DROPOUT = 0.3

BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Current Test Run Timestamp: {CURRENT_TEST_RUN_TIMESTAMP}")
print(f"Loading model checkpoint: {MODEL_CHECKPOINT_PATH}")

# AI prompts:
# Improve formatting
# And sensible outputs when runing the script
# Add control statements for errors
# Note: I ran into several issues so I generated some debug statements using AI


def load_test_features_and_ids(feature_dir_path):
    all_features = []
    all_ids = []
    
    if not os.path.isdir(feature_dir_path):
        print(f"Error: Test feature directory not found: {feature_dir_path}")
        return None, None

    try:
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
        feature_file = f"test_features_saving_batch_{batch_idx}.npy"
        id_file = f"test_ids_saving_batch_{batch_idx}.npy" 
        
        feature_path = os.path.join(feature_dir_path, feature_file)
        ids_path = os.path.join(feature_dir_path, id_file)

        if not os.path.exists(feature_path):
            print(f"Warning: Test feature file not found: {feature_path}. Skipping.")
            continue
        if not os.path.exists(ids_path):
            print(f"Warning: Test ID file not found: {ids_path}. Skipping.")
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
        
    print(f"Loaded {len(all_features)} total test sequences and {len(all_ids)} IDs.")
    return all_features, all_ids


def collate_sequences_test(batch):
    sequences = [item[0] for item in batch] 

    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    lengths = torch.tensor([len(seq) for seq in sequences])
    max_len = sequences_padded.size(1)
    padding_mask = torch.arange(max_len)[None, :] >= lengths[:, None] 

    return sequences_padded, padding_mask


full_test_feature_dir = os.path.join(FEATURE_DIR_BASE, TEST_FEATURE_SUBDIR) 
print(f"Attempting to load test data from: {full_test_feature_dir}")

test_features_all, test_ids_all = load_test_features_and_ids(full_test_feature_dir)

if test_features_all is None or test_ids_all is None:
    print("Exiting due to test data loading issues.")
    exit()

if not test_features_all: 
    print("No test features loaded after load_test_features_and_ids. Exiting.")
    exit()
    
seq_len_example = 0
if isinstance(test_features_all[0], np.ndarray):
    seq_len_example = test_features_all[0].shape[0]
elif isinstance(test_features_all[0], torch.Tensor):
    seq_len_example = test_features_all[0].size(0)
else:
    print(f"Warning: Unexpected feature type for test_features_all[0]: {type(test_features_all[0])}.")

if seq_len_example == 0:
    print(f"Warning: The first feature sequence (test_features_all[0]) has a length of 0 or is of an unexpected type.")
    valid_seq_len_found = False
    for features_item in test_features_all:
        current_item_len = 0
        if isinstance(features_item, np.ndarray) and len(features_item.shape) > 0 : 
            current_item_len = features_item.shape[0]
        elif isinstance(features_item, torch.Tensor) and features_item.dim() > 0: 
            current_item_len = features_item.size(0)
        
        if current_item_len > 0:
            seq_len_example = current_item_len
            valid_seq_len_found = True
            print(f"Using sequence length {seq_len_example} from a subsequent valid feature item for dummy labels.")
            break
else:
    print(f"Using sequence length {seq_len_example} from the first feature item for dummy labels.")


if seq_len_example <= 0: 
    print(f"Final Check Warning: seq_len_example is {seq_len_example}. Forcing to 1 for dummy labels.")
    seq_len_example = 1
    
dummy_frame_labels = np.zeros((len(test_features_all), seq_len_example, 1), dtype=np.float32)
print(f"Created dummy_frame_labels with shape: {dummy_frame_labels.shape}")

test_dataset = AccidentFeatureDataset(test_features_all, dummy_frame_labels) 
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_sequences_test)


model = ViTTransformer(
    feature_dim=VIT_FEATURE_DIM,
    model_dim=MODEL_DIM,
    nhead=N_HEADS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT
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

model.eval() 

all_scores = []

with torch.no_grad():
    for batch_idx, (sequences_padded, padding_mask) in enumerate(test_loader):
        print(f"Processing test batch {batch_idx + 1}/{len(test_loader)}...")
        sequences_padded = sequences_padded.to(device)
        padding_mask = padding_mask.to(device)
        
        _, seq_probs = model(sequences_padded, src_key_padding_mask=padding_mask) 
        
        scores_batch = seq_probs.cpu().numpy()
        if scores_batch.ndim > 1 and scores_batch.shape[1] == 1: 
            scores_batch = scores_batch.squeeze(1)
        all_scores.extend(scores_batch.tolist())

if len(all_scores) != len(test_ids_all):
    print(f"Critical Error: Number of scores ({len(all_scores)}) does not match number of IDs ({len(test_ids_all)}).")
    print("This might be due to issues in data loading or processing. Submission will be incorrect.")
else:
    print(f"Generated {len(all_scores)} scores for {len(test_ids_all)} test videos.")

submission_dir = "submissions"
os.makedirs(submission_dir, exist_ok=True)
submission_filename = f"submission_ViTTransformer_{CURRENT_TEST_RUN_TIMESTAMP}.csv"
submission_path = os.path.join(submission_dir, submission_filename)

with open(submission_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'score']) 
    for vid_id, score in zip(test_ids_all, all_scores): 
        writer.writerow([vid_id, f"{score:.6f}"]) 

print(f"Successfully saved predictions to {submission_path}")
