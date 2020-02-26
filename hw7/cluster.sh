wget https://www.dropbox.com/s/xya84pxbj4zn3ua/hw7_model.h5?dl=1
mv 'hw7_model.h5?dl=1' model.h5
python cluster.py $1 $2 $3
