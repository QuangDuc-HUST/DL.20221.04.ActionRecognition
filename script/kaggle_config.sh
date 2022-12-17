#
# Kaggle config
# 
API='{"username":"alas123dc", "key":"042dab2af1f9d5115f71498bafa92cc3"}'
mkdir -p ~/.kaggle/
echo $API > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
echo "Finish set up API."