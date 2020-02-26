import sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, BatchNormalization, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Dropout, ReLU, ZeroPadding2D
import keras.backend as K
def load_data(path):
    data = pd.read_csv(path)
    m, _ = data.shape
    X = np.zeros((m, 48, 48, 1))
    for i in range(m):
        X[i] = np.array(data['feature'][i].split()).astype(np.float32).reshape(48, 48, 1)
    X = X / 255
    return X
X = load_data(sys.argv[1])
def load_model():
    model = Sequential()
    model.add(Lambda(lambda x: K.repeat_elements(x, 3, 3), input_shape=(48, 48, 1)))
    model.add(Conv2D(32, 3, strides=2, padding='same', use_bias=False))
    model.add(ReLU(max_value=6))
    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(3, padding='same', use_bias=False))
    model.add(ReLU(max_value=6))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 1, padding='same', use_bias=False))
    model.add(ReLU(max_value=6))
    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(3, padding='same', use_bias=False))
    model.add(ReLU(max_value=6))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 1, padding='same', use_bias=False))
    model.add(ReLU(max_value=6))
    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(3, strides=2, padding='same', use_bias=False))
    model.add(ReLU(max_value=6))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 1, padding='same', use_bias=False))
    model.add(ReLU(max_value=6))
    model.add(BatchNormalization())
    for _ in range(3):
        model.add(DepthwiseConv2D(3, padding='same', use_bias=False))
        model.add(ReLU(max_value=6))
        model.add(BatchNormalization())
        model.add(Conv2D(128, 1, padding='same', use_bias=False))
        model.add(ReLU(max_value=6))
        model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation='softmax'))
    return model
model = load_model()
weights = np.load('weights.npz', allow_pickle=True)['weights']
overwrite = list()
for i in weights:
    overwrite.append(i.astype(np.float32))
model.set_weights(overwrite)
y = model.predict(X)
y_pred = np.argmax(y, axis=1)
print("y_predict:", y_pred, "/ shape:", y_pred.shape)
data = np.c_[np.arange(len(y_pred)), y_pred]
fo = open(sys.argv[2], 'w')
fo.write(pd.DataFrame(data, columns=['id', 'label']).to_csv(index=False))
fo.close()