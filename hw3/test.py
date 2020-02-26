import sys
import numpy as np
import pandas as pd
from keras.models import load_model

def load_data(path):
    data = pd.read_csv(path)
    m, _ = data.shape
    X = np.zeros((m, 48, 48, 1))
    for i in range(m):
        X[i] = np.array(data['feature'][i].split()).astype(np.float32).reshape(48, 48, 1)
    X = X / 127.5 - 1
    return X

X = load_data(sys.argv[1])
model = load_model('model.h5')
y = model.predict(X)
y_predict = np.argmax(y, axis=1)
print("y_predict:", y_predict, "/ shape:", y_predict.shape)
data = np.c_[np.arange(len(y_predict)), y_predict]
fo = open(sys.argv[2], 'w')
fo.write(pd.DataFrame(data, columns=['id', 'label']).to_csv(index=False))
fo.close()