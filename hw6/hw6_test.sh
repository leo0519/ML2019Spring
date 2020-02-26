wget https://www.dropbox.com/s/zs3w2g1rmf7vthi/hw6_model.h5?dl=1
mv hw6_model.h5?dl=1 model.h5
wget https://www.dropbox.com/s/st8j3mengqacnf5/hw6_model.wv?dl=1
mv hw6_model.wv?dl=1 model.wv
python test.py $1 $2 $3