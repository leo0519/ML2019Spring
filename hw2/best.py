import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

if __name__ == '__main__':
    X_train = pd.read_csv(sys.argv[1]).values
    y_train = pd.read_csv(sys.argv[2]).values.reshape(-1)
    X_test = pd.read_csv(sys.argv[3]).values
    print("X_train:", X_train.shape, end=' / ')
    print("y_train:", y_train.shape, end=' / ')
    print("X_test:", X_test.shape)
    gbc = GradientBoostingClassifier(n_estimators=700)
    gbc.fit(X_train, y_train)
    y_predict = gbc.predict(X_test)
    print("y_predict:", y_predict, "/ shape:", y_predict.shape)
    data = np.c_[np.arange(len(y_predict)) + 1, y_predict]
    fo = open(sys.argv[4], 'w')
    fo.write(pd.DataFrame(data, columns=['id', 'label']).to_csv(index=False))
    fo.close()