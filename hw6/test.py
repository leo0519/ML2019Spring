import sys
import numpy as np
import pandas as pd
import jieba
from keras.models import load_model
from gensim.models.word2vec import Word2Vec
jieba.load_userdict(sys.argv[2])
word_dim = 100
padding = 50
test_X = pd.read_csv(sys.argv[1])
comments = test_X['comment'].values
words = list()
nComment = len(comments)
for i in range(nComment):
    word = jieba.lcut(comments[i])
    nWord = len(word)
    for j in range(nWord):
        if word[j][0] in ['B', 'b']:
            word[j] = ' '
    for j in range(nWord):
        if j != ' ' and word.count(word[j]) > 5:
            for k in range(j + 1, nWord):
                if word[k] == word[j]:
                    word[k] = ' '
    for j in range(nWord):
        if j != ' ' and word[j] in """！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏""":
            word[j] = ' '
    while ' ' in word:
        word.remove(' ')
    word = word[:padding]
    words.append(word)
    if (i + 1) % 1000 == 0:
        print("[Word Segmentation]", int((i + 1) / nComment * 100), "% ...", end='\n' if i + 1 == nComment else '\r')
wv_model = Word2Vec.load('model.wv')
X_test = np.zeros((nComment, padding, word_dim))
for i in range(nComment):
    for j in range(padding):
        try:
            X_test[i][j] = wv_model.wv.__getitem__(words[i][j])
        except:
            pass
    if (i + 1) % 1000 == 0:
        print("[Word Indexing]", int((i + 1) / nComment * 100), "% ...", end='\n' if i + 1 == nComment else '\r')
model = load_model('model.h5')
y_predict = model.predict(X_test)
y_predict = (y_predict > 0.5).astype(np.int).reshape(-1)
data = np.c_[np.arange(len(y_predict)), y_predict]
fo = open(sys.argv[3], 'w')
fo.write(pd.DataFrame(data, columns=['id', 'label']).to_csv(index=False))
fo.close()