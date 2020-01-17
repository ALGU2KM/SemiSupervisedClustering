import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from keras.utils import np_utils
from DEC import DeepEmbeddingClustering
from sklearn.model_selection import train_test_split
from keras.models import load_model

sca = MinMaxScaler()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

dados = pd.read_csv('d:/basedados/sementes.csv')
#dados = dados[dados['classe'] < 3]
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values

L, U, y, yu = train_test_split(X,Y, train_size=0.05, test_size=0.95, stratify=Y)

dec = DeepEmbeddingClustering(3, np.size(X, axis=1), batch_size=30)
dec.initialize(U)
dec.cluster(U)

Xt = tsne.fit_transform(dec.encoder.predict(U))
preditas = dec.DEC.predict_classes(U)
acuracia = dec.cluster_acc(yu, preditas)

cores = ['#000000', '#0000FF', '#7FFFD4', '#008000', '#CD853F', '#8B008B', '#FF0000','#FFA500', '#FFFF00', '#FF1493']
for i, x in enumerate(Xt):
    plt.scatter(x[0], x[1], c = cores[yu[i]], s = 5)
plt.legend()
plt.show()