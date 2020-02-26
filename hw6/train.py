import sys
import string
import random
import numpy as np
import pandas as pd
import jieba
from datetime import datetime
from gensim.models.word2vec import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, GRU, Bidirectional, TimeDistributed, Flatten
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)
jieba.load_userdict(sys.argv[4])
random.seed(datetime.now())
word_dim = 100
padding = 50
train_X = pd.read_csv(sys.argv[1])
train_y = pd.read_csv(sys.argv[2])
test_X = pd.read_csv(sys.argv[3])
print("[Load Data] Train %d - Test %d" %(train_X.shape[0], test_X.shape[0]))
comments = np.concatenate([train_X['comment'].values, test_X['comment'].values], axis=0)
words = list()
nComment = len(comments)
for i in range(nComment):
    word = jieba.lcut(comments[i])
    nWord = len(word)
    #移除樓層
    for j in range(nWord):
        if word[j][0] in ['B', 'b']:
            word[j] = ' '
    #移除大量相同字
    for j in range(nWord):
        if j != ' ' and word.count(word[j]) > 5:
            for k in range(j + 1, nWord):
                if word[k] == word[j]:
                    word[k] = ' '
    #移除標點符號
    for j in range(nWord):
        if j != ' ' and word[j] in """！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏""":
            word[j] = ' '
    #移除空白
    while ' ' in word:
        word.remove(' ')
    word = word[:padding]
    words.append(word)
    if (i + 1) % 1000 == 0:
        print("[Word Segmentation]", int((i + 1) / nComment * 100), "% ...", end='\n' if i + 1 == nComment else '\r')
words_shuffle = words.copy()
random.shuffle(words_shuffle)
wv_model = Word2Vec(words_shuffle, size=word_dim, min_count=3, iter=10, workers=6, sg=1)
wv_items = wv_model.wv.vocab.items()
print("[# of Items]", len(wv_items))
for i in words_shuffle[0][:10]:
    try:
        print(i, wv_model.wv.most_similar(i, topn=5))
    except:
        print(i, "Not found in vocabularies.")
X = np.zeros((nComment, padding, word_dim))
for i in range(nComment):
    for j in range(padding):
        try:
            X[i][j] = wv_model.wv.__getitem__(words[i][j])
        except:
            pass
    if (i + 1) % 1000 == 0:
        print("[Word Indexing]", int((i + 1) / nComment * 100), "% ...", end='\n' if i + 1 == nComment else '\r')
X_train = X[:train_X.shape[0]]
X_test = X[train_X.shape[0]:]
y_train = train_y['label'].values
RNN_model = Sequential()
RNN_model.add(LSTM(256, return_sequences=True, input_shape=(padding, word_dim)))
RNN_model.add(LSTM(256))
RNN_model.add(Dense(256))
RNN_model.add(Dense(256))
RNN_model.add(Dense(1, activation='sigmoid'))
RNN_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
RNN_model.summary()
modelCheckpoint = ModelCheckpoint('rnn.{epoch:02d}-{val_acc:.4f}.h5', monitor='val_acc', save_best_only=True)
csv_logger = CSVLogger('rnn.log')
RNN_model.fit(X_train, y_train, batch_size=1024, epochs=50, validation_split=0.14, callbacks=[modelCheckpoint, csv_logger])