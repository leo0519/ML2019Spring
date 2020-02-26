import sys
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, BatchNormalization, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Dropout, ReLU, ZeroPadding2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K
def load_data(path):
    data = pd.read_csv(path)
    m, _ = data.shape
    X = np.zeros((m, 48, 48, 1))
    for i in range(m):
        X[i] = np.array(data['feature'][i].split()).astype(np.float32).reshape(48, 48, 1)
    X = X / 255
    y = to_categorical(data['label'].values, 7)
    return X, y
X, y = load_data(sys.argv[1])
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]
X_train, X_val = X[2800:], X[:2800]
y_train, y_val = y[2800:], y[:2800]
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
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
dataGen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
dataGen.fit(X_train)
modelCheckpoint = ModelCheckpoint('model.h5', monitor='val_acc', save_best_only=True, save_weights_only=True)
model.fit_generator(dataGen.flow(X_train, y_train, 128), steps_per_epoch=len(X_train)*8//128, epochs=20, validation_data=(X_val, y_val), callbacks=[modelCheckpoint])
model.load_weights('model.h5')
weights = list()
for i in model.get_weights():
    weights.append(i.astype(np.float16))
np.savez_compressed('weights', weights=weights)