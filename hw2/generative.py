#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd

class probabilistic_generative():
    def __init__(self):
        pass
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def __scale(self, X, isTrain=True):
        if isTrain:
            self.__mean = np.mean(X, axis=0)
            self.__std = np.std(X, axis=0)
        return (X - self.__mean) / (self.__std + 1e-10)
    def fit(self, X, y):
        m, n = X.shape
        X = self.__scale(X)
        prop, mean = np.zeros(2), np.zeros((2, n))
        cov_sum = 0
        for i in range(2):
            X_class = X[y == i]
            prop[i] = len(X_class) / m
            mean[i] = np.mean(X_class, axis=0)
            cov = np.mean([j.reshape(-1, 1).dot(j.reshape(1, -1)) for j in (X_class - mean[i])], axis=0)
            cov_sum += prop[i] * cov
        inv_cov = np.linalg.inv(cov_sum)
        self.weight = (mean[0] - mean[1]).dot(inv_cov)
        self.bias = sum([(i - 0.5) * mean[i].dot(inv_cov).dot(mean[i]) for i in range(2)]) + np.log(prop[0] / prop[1] + 1e-10)
        accuracy = (self.predict(X, scaled=True) == y).sum() / len(y)
        print("acc: %.3f" %accuracy)
    def predict(self, X, scaled=False):
        if not scaled:
            X = self.__scale(X, isTrain=False)
        return (self.__sigmoid(self.weight.dot(X.T) + self.bias) < 0.5).astype(np.int32)

if __name__ == '__main__':
    X_train = pd.read_csv(sys.argv[1]).values
    y_train = pd.read_csv(sys.argv[2]).values.reshape(-1)
    X_test = pd.read_csv(sys.argv[3]).values
    print("X_train:", X_train.shape, end=' / ')
    print("y_train:", y_train.shape, end=' / ')
    print("X_test:", X_test.shape)
    pg = probabilistic_generative()
    pg.fit(X_train, y_train)
    y_predict = pg.predict(X_test)
    print("y_predict:", y_predict, "/ shape:", y_predict.shape)
    data = np.c_[np.arange(len(y_predict)) + 1, y_predict]
    fo = open(sys.argv[4], 'w')
    fo.write(pd.DataFrame(data, columns=['id', 'label']).to_csv(index=False))
    fo.close()