# NNDL_Project 

Order to train and test the models:

### Training

#### ResNet-LSTM Model
1. First, run ```run_feature_extract_batched.py```. Your dataset needs to be in the same directory named as is done in the program. This program extracts video frames at a sub-sampled rate from each video file and extracts its features from the ResNet18 backbone. The extracted features are then stored in ```ResNet_Features/train_features_{TIMESTAMP}.npy```. These features will be used in the sequential model. 
2. Then, run ```train_seq_model.py```. This uses a Bi-LSTM architecture on the sequence of video features extracted in Step 1. The output model is stored in ```checkpoints/ResNetLSTM_best_{MODEL_TIMESTAMP}```. 

#### ViT-Transformer Model
1. First, run ```run_gpu_feature_extraction.py``` with CLIP ViT backbone. This will extract features using the CLIP ViT model and store them in ```processed_data/CLIP_ViT_Features_clip-vit-large-patch14/run_{TIMESTAMP}```.
2. Then, run ```train_transformer_vit.py```. This uses a Transformer architecture on the sequence of video features. The output model is stored in ```checkpoints/ViTTransformer_best_{TIMESTAMP}```.

#### TimeSformer Model
1. Run ```train_timesformer.py``` directly. This model processes video frames directly without requiring a separate feature extraction step. The output model is stored in ```checkpoints/model_best.pth```.

#### XCLIP Model
1. Run ```train_xclip.py``` directly. This model also processes video frames directly and includes both vision and text encoders. The output model is stored in ```checkpoints/model_best.pth```.

### Evaluation
1. For ResNet-LSTM and ViT-Transformer:
   - First, run the ```run_test_feature_extract_batched.py```. Similar to Training Step 1.
   - Then, run ```test_seq_model.py``` or ```test_transformer_vit.py``` respectively. As the model timestamp and testing timestamps would be different, change both of them accordingly.

2. For TimeSformer and XCLIP:
   - Run ```test_timesformer.py``` or ```test_xclip.py``` directly.

You will then have an output csv file in the ```submissions/submission_{timestamp}.csv```. Submit this to the kaggle competition to view results. 

### Note: 
Please use the batched versions of the models to prevent CPU & RAM from going out of memory and terminating without warning. 

### Model Architectures
1. **ResNet-LSTM**: Uses ResNet18 for feature extraction followed by a Bi-LSTM for sequence modeling
2. **ViT-Transformer**: Uses CLIP ViT for feature extraction followed by a Transformer encoder
3. **TimeSformer**: End-to-end video transformer model that processes video frames directly
4. **XCLIP**: Multimodal model that combines vision and text encoders for video understanding

