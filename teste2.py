import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from keras.utils import np_utils
from DEC import DeepEmbeddingClustering
from sklearn.model_selection import train_test_split
from keras.models import load_model

k = 3

sca = MinMaxScaler()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

dados = pd.read_csv('d:/basedados/mnist64.csv')
#dados = dados[dados['classe'] < 3]
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values

#L, U, y, yu = train_test_split(X,Y, train_size=0.05, test_size=0.95, stratify=Y)

#INICIALIZAÇÃO DO DEC
dec = DeepEmbeddingClustering(k, np.size(X, axis=1), batch_size=30)
dec.initialize(X)
dec.cluster(X)
#G = pd.DataFrame(U)
#G['g'] = dec.DEC.predict_classes(U)

#GRÁFICO DO AGRUPAMENTO
Xt = tsne.fit_transform(dec.encoder.predict(U))
preditas = dec.DEC.predict_classes(U)
cores = ['red','blue', 'green']

for i, x in enumerate(Xt):
    plt.scatter(x[0], x[1], c = cores[preditas[i]], s = 5)
plt.legend()
plt.show()

#ROTULAÇAO
Lz = dec.encoder.predict(L)                         #Calcula L no espaço Z
GL = pd.DataFrame(L)
GL['g'] = dec.DEC.predict_classes(L)
GL['y'] = y

def criar_centroides(L):
    classes = L['y'].unique()
    #L = L.drop(['g'], axis=1)
    centroides = L.groupby(['y']).mean()
    return (centroides.values, classes)

GL2 = GL[GL['g']==2]
G2 = G[G['g']==2]

GL2 = GL2.drop(['g'], axis=1)         
G2 = G2.drop(['g'], axis=1)

pg = dec.DEC.predict(G2.values)                  #Calcula distribuição de probabilidade dos elementos do grupo
centroides,classes = criar_centroides(GL2)    #Calcula os centroides de cada classe
centros = dec.DEC.predict(centroides)           #calcula distribuição de probabilidade de cada centro



















