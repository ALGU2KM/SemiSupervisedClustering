# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from Labeling import DeepSelfLabeling

sca = MinMaxScaler()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

dados = pd.read_csv('d:/basedados/mnist64.csv')
#dados = dados[dados['classe'] < 3]
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values

z = np.size(np.unique(Y))

DSL = DeepSelfLabeling(k=z,dim=np.size(X, axis=1), lote=128)
DSL.inicializacao(X, epocas=2000)
preditas = np.argmax(DSL.DEC.predict(X),1)

q = DSL.DEC.predict(X)
p = DSL.p_mat(DSL.DEC.predict(X))

DSL.DEC.fit(X, p, epochs=1000)

Xt = tsne.fit_transform(DSL.encoder.predict(X))
cores = ['#000000', '#0000FF', '#7FFFD4', '#008000', '#CD853F', '#8B008B', '#FF0000','#FFA500', '#FFFF00', '#FF1493']
for i, x in enumerate(Xt):
    plt.scatter(x[0], x[1], c = cores[Y[i]], s = 5)
plt.legend()
plt.show()

respostaZ = DSL.encoder.predict(X)
encoded_imgs = DSL.autoencoder.predict(X)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X[i].reshape(8, 8))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(respostaZ[i].reshape(1, 10).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + n*2)
    plt.imshow(encoded_imgs[i].reshape(8, 8))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)