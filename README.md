# Deep Learning Project
# Action Recognition - Group 04 - DSAI K65 - HUST 
*The project structure is inspired by [cs230-stanford course](https://github.com/cs230-stanford/cs230-code-examples).*

**Note: All the code need to run in this directory.* 

*All the code should be run on one of three platforms Linux OS, Google Colab, and Kaggle.*

---
## Project Structure 

```
README.md               
ckp/                    # Default checkpoint folder
data/                   # Default data folder
model/
script/                 # Script for downloading the extracted data
build_dataset.py        
evaluate.py
inference.py
train.py
utils.py
```
*For more details, please read the README.md in each sub-directory and each file.*

---
# Code Execution   
Our project has two versions: *Linux Environment* and *Google Colab environment* (basically they are the same). 

However, we highly recommend to use Google Colab version because of its convenience (our dataset is from 20GB to 200GB after extracting frames from video)

## Requirements
### Google Colab Version (recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/drive/folders/1_L19tnAa6udACh7cNzx7mI6-KUYZReG_?usp=sharing)

*We used Google Colab Pro Version for this project as some problems may be occurred if you training on the large ( >100GB ) extracted frame dataset.*

Please follow the instructions on the Colab Version notebook
### Linux Environment ([Miniconda](https://docs.conda.io/en/latest/miniconda.html))
```
    git clone https://github.com/QuangDuc-HUST/DL.20221.04.ActionRecognition
    cd DL.20221.04.ActionRecognition/

    conda env create -f environment.yml
    conda activate dlenv
```
---


## Quick starts (on your local machine)
*For more details, we also comment it in every files*


#### Build the simple `dataset` (download extracted frames data)
~ 20GB for the basic extracted frames dataset (for more dataset, please read in `build_dataset.py`)
```
    python build_dataset --dataset ucf101 --process_type 5_frames_uniform --kaggle
```
* This will automatic download the frame extracted (5 frames in one video uniform distributed) dataset UCF101 from kaggle platform, then unzip and store in `./data/UCF101/5_frames_uniform/`.

#### `Training` the model. Simple run
```
   python train.py --model_name late_fusion --batch_size 64 --data_dir './data/UCF101/5_frames_uniform/' --dataset 'ucf101' --max_epochs 20 --lr 0.0005 
```
* We train the downloaded extracted frame dataset by using Late Fusion model with batch size = 64, learning rate = 0.0005, maximum epochs = 20.
#### `Evaluate` the model on the test set
```
    python evaluate.py --model_name late_fusion --batch_size 32 --data_dir './data/UCF101/5_frames_uniform/' --dataset 'ucf101' 
```
* We evaluate the ucf101 on the trained previous model with batch size is 32.

#### `Prediction` the model on a new video 
```
    python inference.py --model_name late_fusion --video_path /path/to/video --dataset ucf101 
```
* Predict a new video based on ucf101 dataset label and return a list of percentage of classes


## Advanced training use

We recommend read through `train.py` and `build_dataset` to get intuition of what options we offer.

---
## EDA  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c594erS-_glCHjxHIWpV1kh2A_cOf6ti?usp=sharing)


## Visualisation of what the model has learned
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BG4_e7xbXrhvO9rffRuH2pUtj_sxppdf?usp=sharing)

## Deployment on Colab Machine
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Cm7VICYWw32mVDEAEXYDsUQHrw3TlKfN?usp=sharing)


