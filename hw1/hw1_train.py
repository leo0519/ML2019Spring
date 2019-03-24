import sys, math
import numpy as np
import pandas as pd

def load_train_data():
    data = pd.read_csv(sys.argv[1], encoding='big5')
    X = data.drop(['日期', '測站', '測項'], axis=1).applymap(lambda x: 0 if x == 'NR' else x).values.astype(np.float32).reshape(-1, 18, 24).transpose((0, 2, 1)).reshape(-1, 18)
    X_stack = np.empty((len(X) - 9, 18 * 9))
    for i in range(len(X) - 9):
        X_stack[i] = X[i: i + 9].reshape(-1)
    y = data.loc[data['測項'] == 'PM2.5'].drop(['日期', '測站', '測項'], axis=1).values.astype(np.float32).reshape(-1)[9:]
    return X_stack, y

class gradient_descent:
    def __init__(self):
        pass
    def train(self, X, y, epochs=10000, learning_rate=0.01, batch_size=128):
        m, n = X.shape
        self.theta = np.zeros(n + 1)
        ada = np.zeros(n + 1)
        X = np.c_[np.ones(m), X]
        for epoch in range(epochs):
            idx = np.random.permutation(len(X))
            X_shuffle = X[idx]
            y_shuffle = y[idx]
            for i in range(math.floor(len(X) / batch_size)):
                X_split = X_shuffle[i * batch_size: (i + 1) * batch_size]
                y_split = y_shuffle[i * batch_size: (i + 1) * batch_size]
                gradient = 2 * X_split.T.dot(X_split.dot(self.theta) - y_split)
                ada += gradient ** 2
                self.theta = self.theta - learning_rate * gradient / np.sqrt(ada)
    def predict(self, X):
        X = np.c_[np.ones(len(X)), X]
        return X.dot(self.theta)

X_train, y_train = load_train_data()
gd = gradient_descent()
gd.train(X_train, y_train)
np.save('model.npy', gd.theta)