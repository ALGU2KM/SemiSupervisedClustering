import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from keras.utils import np_utils
from Labeling import DeepSelfLabeling
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import accuracy_score

sca = MinMaxScaler()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

dados = pd.read_csv('d:/basedados/sementes.csv')
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values

L, U, y, yu = train_test_split(X,Y, train_size=0.05, test_size=0.95, stratify=Y)

dsl = DeepSelfLabeling(k=np.size(np.unique(Y)), dim=np.size(L, axis=1),lote=30)

PL, PU = dsl.self_Labeled(L, U, y)

Rot, NaoRot = dsl.divisao_grupos(PU, PL)

G1 = pd.DataFrame(np.array([
        [1,10,0],
        [2,10,0],
        [2,30,0 ],]), columns=['cor','tamanho', 'textura'])

G2 = pd.DataFrame(np.array([
        [1,20,1],
        [3,20,1],
        [2,20,1]]), columns=['cor','tamanho', 'textura'])

G3 = pd.DataFrame(np.array([
        [4,30,1],
        [4,20,1]]), columns=['cor','tamanho', 'textura'])

def entropia(X):
    T = np.size(X, axis=1)
    colunas = X.columns.values
    H = []
    for c in colunas:
        dados = X[c]
        p = dados.value_counts()
        p = p.values / T
        p = p * np.log2(p)
        H.append(p.sum())
    return H

H = entropia(G1)
        