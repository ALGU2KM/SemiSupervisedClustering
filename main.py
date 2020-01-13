import pandas as pd
import numpy as np
from Clustering import SemiClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from keras.utils import np_utils
from DEC import DeepEmbeddingClustering

sca = MinMaxScaler()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

dados = pd.read_csv('d:/basedados/mnist64.csv')
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values

dec = DeepEmbeddingClustering(10, np.size(X, axis=1))

dec.initialize(X)

#sdc = SemiClustering(10, np.size(X, 1))
#sdc.treinar(X, 300)
#sdc.cluster(X)
#y_treino = sdc.encoder.predict(X)
#y_treino = sdc.p_mat(y_treino)
#y_treino = np.argmax(y_treino, axis=1)
#y_treino = np_utils.to_categorical(y_treino)

#sdc.DEC.fit(X, y_treino, epochs=100)

dec.cluster(X)

Xt = tsne.fit_transform(X)
preditas = dec.DEC.predict_classes(X)
acuracia = dec.cluster_acc(Y, preditas)

cores = ['#000000', '#0000FF', '#7FFFD4', '#008000', '#CD853F', '#8B008B', '#FF0000','#FFA500', '#FFFF00', '#FF1493']
for i, x in enumerate(Xt):
    plt.scatter(x[0], x[1], c = cores[preditas[i]], s = 5)
plt.legend()
plt.show()