import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import load_model
from keras.utils import to_categorical
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.color import rgb2gray
from lime.wrappers.scikit_image import SegmentationAlgorithm

def load_data(path):
    data = pd.read_csv(path)
    m, _ = data.shape
    X = np.zeros((m, 48, 48, 1))
    for i in range(m):
        X[i] = np.array(data['feature'][i].split()).astype(np.float32).reshape(48, 48, 1)
    X = X / 127.5 - 1
    y = to_categorical(data['label'].values, 7)
    return X, y

def saliency_map(model, X, y, num_class=7):
    y_label = np.argmax(y, axis=1).tolist()
    y_idx = [0, 299, 2, 7, 3, 15, 4]
    for sel in y_idx:
        softmax = model.predict(X[sel].reshape(-1, 48, 48, 1))
        label = softmax.argmax(axis=-1)
        inputs = model.input
        outputs = model.output[:, label[0]]
        grad = K.gradients(outputs, inputs)[0]
        sal = K.function([inputs, K.learning_phase()], [grad])
        sal_grad = sal([X[sel].reshape(-1, 48, 48, 1), 0])[0].reshape(48, 48, 1)
        sal_grad = 0.2 * (sal_grad - np.mean(sal_grad)) / (np.std(sal_grad) + 1e-10)
        sal_grad = np.clip(sal_grad, -1, 1)
        heat = sal_grad.reshape(48, 48)
        hm = plt.imshow(heat, cmap='jet')
        plt.colorbar(hm)
        plt.savefig(os.path.join(sys.argv[2], 'fig1_' + str(y_label[sel]) + '.jpg'))
        plt.close('all')

def filters(model):
    dictLayer = dict([layer.name, layer] for layer in model.layers)
    listCollectLayers = dictLayer['conv2d_1'].output
    plt.figure(figsize=(16, 16))
    for i in range(64):
        inputs = model.input
        X = np.random.random((1, 48, 48, 1)) * 2 - 1
        y = listCollectLayers[:, :, :, i]
        mean = K.mean(K.abs(y))
        grad = K.gradients(mean, inputs)[0]
        norm = grad / (K.sqrt(K.mean(K.square(grad))) + 1e-10)
        func = K.function([inputs, K.learning_phase()], [y, norm])
        for j in range(100):
            loss, gradients = func([X, 0])
            X += gradients * 0.02
        plt.subplot(8, 8, i + 1)
        plt.imshow(X.reshape(48, 48) / 2 + 0.5, cmap='Oranges')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.savefig(os.path.join(sys.argv[2], 'fig2_1.jpg'))
    plt.close('all')

def results(model, X):
    X = X[123].reshape(1, 48, 48, 1)
    inputs = model.input
    dictLayer = dict([layer.name, layer] for layer in model.layers)
    listCollectLayers = dictLayer['conv2d_1'].output
    func = K.function([inputs, K.learning_phase()], [listCollectLayers])
    y = func([X, 0])[0]
    plt.figure(figsize=(16, 16))
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.imshow(y[:, :, :, i].reshape(48, 48) / 2 + 0.5, cmap='Oranges')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.savefig(os.path.join(sys.argv[2], 'fig2_2.jpg'))
    plt.close('all')

def lime(model, X, y):
    idx = [0, 299, 2, 7, 3, 15, 4]
    explainer = lime_image.LimeImageExplainer()
    segmenter = SegmentationAlgorithm('slic', n_segments=100, compactness=1, sigma=1)
    for i, j in enumerate(idx):
        explanation = explainer.explain_instance(X[j].reshape(48, 48), lambda x: model.predict(rgb2gray(x).reshape(-1, 48, 48, 1)), labels=(i,), top_labels=7, hide_color=0, num_samples=1000, segmentation_fn=segmenter)
        temp, mask = explanation.get_image_and_mask(i, positive_only=False, num_features=10, hide_rest=False)
        plt.imshow(temp / 2 + 0.5)
        plt.savefig(os.path.join(sys.argv[2], 'fig3_' + str(i) + '.jpg'))
        plt.close('all')

X, y = load_data(sys.argv[1])
model = load_model('model.h5')
saliency_map(model, X, y)
filters(model)
results(model, X)
lime(model, X, y)