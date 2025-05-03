# NNDL_Project 

Order to train and test the model:

### Training
1. First, run ```run_feature_extract_batched.py```. Your dataset needs to be in the same directory named as is done in the program. This program extracts video frames at a sub-sampled rate from each video file and extracts its features from the ResNet18 backbone. The extracted features are then stored in ```ResNet_Features/train_features_{TIMESTAMP}.npy```. These features will be used in the sequential model. 
2. Then, run ```train_seq_model.py```. This uses an Bi-LSTM architecture on the sequence of video features extracted in Step 1. The output model is stored in ```checkpoints/ResNetLSTM_best_{MODEL_TIMESTAMP}```. 

### Evaluation
1. First, run the ```run_test_feature_extract_batched.py```. Similar to Training Step 1.
2. Then, run ```test_seq_model.py```. As the model timestamp and testing timestamps would be different, change both of them accordingly. You will then have an output csv file in the ```submissions/submission_{timestamp}.csv```. Submit this to the kaggle competition to view results. 

### Note: 

Please use the batched versions of the models to prevent CPU & RAM from going out of memory and terminating without warning. 