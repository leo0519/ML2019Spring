import sys, numpy as np, pandas as pd

def load_test_data():
    data = pd.read_csv(sys.argv[1], header=None)
    X = data.drop([0, 1], axis=1).applymap(lambda x: 0 if x == 'NR' else x).values.astype(np.float32).reshape(-1, 18, 9).transpose((0, 2, 1)).reshape(-1, 162)
    return X

def load_predict(y):
    data = np.c_[['id_' + str(i) for i in range(len(y))], y]
    data = pd.DataFrame(data=data, columns=['id', 'value'])
    fo = open(sys.argv[2], "w")
    fo.write(data.to_csv(index=False))
    fo.close()
    
w = np.load('model.npy')
X = load_test_data()
X_bias = np.c_[np.ones(len(X)), X]
load_predict(X_bias.dot(w))