import sys
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Flatten, Activation, MaxPooling2D, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

def load_data(path):
    data = pd.read_csv(path)
    m, _ = data.shape
    X = np.zeros((m, 48, 48, 1))
    for i in range(m):
        X[i] = np.array(data['feature'][i].split()).astype(np.float32).reshape(48, 48, 1)
    X = X / 127.5 - 1
    y = to_categorical(data['label'].values, 7)
    return X, y

def create_model():
    i = Input(shape=(48, 48, 1))
    j = Conv2D(64, 5, padding='same')(i)
    j = LeakyReLU(alpha=0.05)(j)
    j = BatchNormalization()(j)
    j = MaxPooling2D(2, padding='same')(j)
    j = Dropout(0.2)(j)
    j = Conv2D(128, 3, padding='same')(j)
    j = LeakyReLU(alpha=0.05)(j)
    j = BatchNormalization()(j)
    j = MaxPooling2D(2, padding='same')(j)
    j = Dropout(0.3)(j)
    j = Conv2D(512, 3, padding='same')(j)
    j = LeakyReLU(alpha=0.05)(j)
    j = BatchNormalization()(j)
    j = MaxPooling2D(2, padding='same')(j)
    j = Dropout(0.4)(j)
    j = Conv2D(512, 3, padding='same')(j)
    j = LeakyReLU(alpha=0.05)(j)
    j = BatchNormalization()(j)
    j = MaxPooling2D(2, padding='same')(j)
    j = Dropout(0.4)(j)
    j = Flatten()(j)
    j = Dense(512, activation='relu')(j)
    j = BatchNormalization()(j)
    j = Dropout(0.5)(j)
    j = Dense(512, activation='relu')(j)
    j = BatchNormalization()(j)
    j = Dropout(0.5)(j)
    j = Dense(7, activation='softmax')(j)
    model = Model(inputs=i, outputs=j)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    X, y = load_data(sys.argv[1])
    print('data description - X: %d images of size %d x %d x %d - y: %d labels for %d sentiments' % (X.shape + y.shape))
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    X_train, X_val = X[7000:], X[:7000]
    y_train, y_val = y[7000:], y[:7000]
    dataGen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    dataGen.fit(X_train)
    modelCheckpoint = ModelCheckpoint('model.h5', monitor='val_acc', save_best_only=True)
    model = create_model()
    model.fit_generator(dataGen.flow(X_train, y_train, batch_size=128), steps_per_epoch=len(X)//128, epochs=500, validation_data=(X_val, y_val), callbacks=[modelCheckpoint])