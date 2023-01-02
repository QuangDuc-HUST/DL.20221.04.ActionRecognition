# Deep Learning Project
# Action Recognition - Group 04 - DSAI K65 - HUST 
*The project structure is inspired by [cs230-stanford course](https://github.com/cs230-stanford/cs230-code-examples).*

**Note: All the code need to run in this directory.* 

*All the code should be run on one of three platforms Linux OS, Google Colab, and Kaggle.*

---
## Project Structure 

```
├── __pycache__
├── app
│   ├── __pycache__
│   ├── core
│   │   ├── api
│   │   │   └── __pycache__
│   │   ├── constants
│   │   │   └── __pycache__
│   │   └── utils
│   │       └── __pycache__
│   ├── staging
│   │   └── video
│   └── templates
│       └── static
│           ├── css
│           ├── img
│           └── js
├── ckp
├── data
│   ├── HMDB51
│   │   └── annotation
│   └── UCF101
│       └── annotation
├── model
│   ├── __pycache__
│   ├── utils
│   │   └── __pycache__
│   └── weights
└── script
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
    # then conda activate the installed environment
```
---


## Quick starts (on your local machine)
*For more details, we also comment it in every files*


#### Start webserver (FastAPI) in your local machine
```
    uvicorn main:app --reload
```

#### On your browser, go to <a href="http://127.0.0.1:8000" class="external-link" target="_blank">127.0.0.1:8000</a>

#### Chose model type, upload the video (use video from UCF101 dataset only) that you want to make prediction and get the result!!!



