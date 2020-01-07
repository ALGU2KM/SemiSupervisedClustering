import pandas as pd
import numpy as np
from Clustering import SemiClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from keras.utils import np_utils

sca = MinMaxScaler()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

dados = pd.read_csv('d:/basedados/mnist.csv')
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values


sdc = SemiClustering(10, np.size(X, 1))
sdc.treinar(X, 300)
y_treino = sdc.encoder.predict(X)
y_treino = sdc.p_mat(y_treino)
y_treino = np.argmax(y_treino, axis=1)
y_treino = np_utils.to_categorical(y_treino)

sdc.DEC.fit(X, y_treino, epochs=100)

preditas = sdc.DEC.predict_classes(X)

Xt = tsne.fit_transform(X)

plt.scatter(Xt[:,0], Xt[:,1], c = preditas)
plt.legend()
plt.show()