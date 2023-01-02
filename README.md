# Deep Learning Project
# Action Recognition - Group 04 - DSAI K65 - HUST (Two dataset training version)

*Generally, the API is the same for the training one dataset version (the main version) but with some following modifications*
---

### Google Colab Version 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1clNeKs8McY2DIAGWTBJosmwoArQEOKWJ?usp=sharing)

## Quick starts
*For more details, we also comment in every files*


#### Build the simple `dataset` (download extracted frames data)
~ 20GB for the basic extracted frames dataset (for more dataset, please read in `build_dataset.py`)
```
    python build_dataset.py --dataset ucf101 --process_type 5_frames_uniform --kaggle

    python build_dataset.py --dataset hmdb51 --process_type 5_frames_uniform --kaggle
```
* This will automatic download the frame extracted (5 frames in one video uniform distributed) dataset UCF101 from kaggle platform, then unzip and store in `./data/UCF101/5_frames_uniform/`.

#### `Training` the model. Simple run
```
   python train_2_dataset.py --model_name late_fusion --batch_size 64 --max_epochs 40 --data_dir_1 './data/UCF101/5_frames_uniform/' --dataset_1 'ucf101' --data_dir_2 './data/HMDB51/5_frames_uniform/' --dataset_2 'hmdb51' --clip_per_video 1  --lr 0.0005  
```
* We train the downloaded extracted frame dataset by using Late Fusion model with batch size = 64, learning rate = 0.0005, maximum epochs = 40.

#### `Evaluate` the model on the test set
```
    python evaluate_2_dataset.py --model_name late_fusion --batch_size 32 --data_dir_1 './data/UCF101/5_frames_uniform/' --dataset_1 'ucf101' --data_dir_2 './data/HMDB51/5_frames_uniform/' --dataset_2 'hmdb51'  --clip_per_video 1   
```
* Evaluate the trained model on the test set of two datasets.

#### `Inference` the model on the new video
```
    python inference_2_dataset.py --model_name late_fusion --video_path /path/to/video --first_dataset
```

* Inference on a user upload video of the first dataset label (in here it's UCF101).


## For more detail usage, please visit each python file.

