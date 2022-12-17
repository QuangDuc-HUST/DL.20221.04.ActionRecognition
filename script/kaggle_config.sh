#
# Kaggle config
# 
API='{"username":"quangduc0703", "key":"e7ad2c67a6c5798f47aa8f98a865066e"}'
mkdir -p '~/.kaggle/'
echo $API > '~/.kaggle/kaggle.json'
chmod 600 '~/.kaggle/kaggle.json'
echo "Finish set up API."