import sys
import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import keras.backend as K

num_class = 1000
num_img = 200
labels = np.load('labels.npy')
model = load_model('hw5_model.h5')

def noise(model, img, label, epochs=5):
    X = np.copy(img)
    X = X.reshape((1,) + X.shape)
    inputs = model.input
    outputs = model.output[:, label]
    grad = K.gradients(outputs, inputs)[0]
    sign = K.sign(grad)
    func = K.function([inputs], [outputs, sign])
    for epoch in range(epochs):
        X_prep = np.copy(X)
        softmax, gradients = func([preprocess_input(X_prep)])
        X -= gradients
        X = np.clip(X, 0, 255)
    print("[Softmax]", softmax[0])
    return X[0]

for idx, label in enumerate(labels):
    print("[Image ID]", idx)
    img = Image.open(os.path.join(sys.argv[1], '%03d.png' % idx))
    img = np.asarray(img).astype(np.float32)
    adv = noise(model, img, label)
    print("[L-infinity]", np.abs(adv - img).max())
    adv = Image.fromarray(adv.astype(np.uint8))
    adv.save(os.path.join(sys.argv[2], '%03d.png' % idx))
    print()
