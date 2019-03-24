#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd

class logistic_regression():
    def __init__(self):
        pass
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def __scale(self, X, isTrain=True):
        if isTrain:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        return (X - self.mean) / (self.std + 1e-10)
    def fit(self, X, y, epochs=1000, lr=0.1, val=None):
        m, n = X.shape
        self.theta = np.zeros(n+1)
        ada = np.zeros(n+1)
        X = np.concatenate((np.ones((m, 1)), self.__scale(X)), axis=1)
        for epoch in range(epochs+1):
            predict = self.__sigmoid(X.dot(self.theta))
            grad = (predict - y).T.dot(X)
            ada += grad ** 2
            self.theta -= lr / np.sqrt(ada) * grad
            loss = -np.mean(y * np.log(predict + 1e-10) + (1 - y) * np.log(1 - predict + 1e-10))
            acc = ((predict < 0.5) == y).sum() / m
            if epoch % 100 == 0:
                print("Epoch %d/%d" %(epoch,epochs))
                print("loss: %.3f" %loss, "- acc: %.3f" %acc)
    def predict(self, X):
        X = np.concatenate((np.ones((len(X), 1)), self.__scale(X, isTrain=False)), axis=1)
        return (self.__sigmoid(X.dot(self.theta)) >= 0.5).astype(np.int32)

if __name__ == '__main__':
    X_train = pd.read_csv(sys.argv[1]).values
    y_train = pd.read_csv(sys.argv[2]).values.reshape(-1)
    X_test = pd.read_csv(sys.argv[3]).values
    print("X_train:", X_train.shape, end=' / ')
    print("y_train:", y_train.shape, end=' / ')
    print("X_test:", X_test.shape)
    lr = logistic_regression()
    lr.fit(X_train, y_train)
    y_predict = lr.predict(X_test)
    print("y_predict:", y_predict, "/ shape:", y_predict.shape)
    data = np.c_[np.arange(len(y_predict)) + 1, y_predict]
    fo = open(sys.argv[4], 'w')
    fo.write(pd.DataFrame(data, columns=['id', 'label']).to_csv(index=False))
    fo.close()