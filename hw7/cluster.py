import os
import sys
import numpy as np
import pandas as pd
from skimage import io
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Reshape, AveragePooling2D, UpSampling3D, MaxPooling2D, Dense
from keras.losses import mean_absolute_error
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.models import load_model
img = np.zeros((40000, 32, 32, 3))
for i in range(40000):
    img[i] = io.imread(os.path.join(sys.argv[1], '%06d.jpg' % (i + 1))).astype(np.float32)
encoder = load_model('model.h5')
X_AE = encoder.predict(img)
U, sigma, V = np.linalg.svd(X_AE, full_matrices=False)
X_pred = U[:, :512]
kMeans = KMeans(n_clusters=2, random_state=0, n_init=100).fit(X_pred)
X_labels = kMeans.labels_
test_X = pd.read_csv(sys.argv[2])
X_test = test_X.drop(['id'], axis=1).values - 1
m = X_test.shape[0]
y_pred = np.zeros(m)
for i in range(m):
    y_pred[i] = X_labels[X_test[i][0]] == X_labels[X_test[i][1]]
data = np.c_[np.arange(len(y_pred)), y_pred.astype(np.int32)]
fo = open(sys.argv[3], 'w')
fo.write(pd.DataFrame(data, columns=['id', 'label']).to_csv(index=False))
fo.close()