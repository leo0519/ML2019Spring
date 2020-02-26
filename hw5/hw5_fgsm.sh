wget https://www.dropbox.com/s/acm1e7qoul2f3a2/hw5_model.h5?dl=1
mv hw5_model.h5?dl=1 hw5_model.h5
python fgsm.py $1 $2
