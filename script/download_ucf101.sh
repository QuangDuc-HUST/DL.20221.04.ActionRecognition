# Download UCF101 dataset from kaggle
# Working for Linux only

USER_KAGGLE = "quangduc0703"
KAGGLE_TOKEN = "e7ad2c67a6c5798f47aa8f98a865066e"

KAGGLE_CONFIG = 

!mkdir -p /root/.kaggle/



wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar -P data/raw/UCF101/
wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip-P data/raw/UCF101/


