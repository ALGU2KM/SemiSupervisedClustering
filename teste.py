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
import keras.backend as K
from scipy.stats import t
from scipy.stats import norm

sca = MinMaxScaler()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

dados = pd.read_csv('d:/basedados/sementes.csv')
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values
dados = pd.DataFrame(X)
dados['classe'] = Y


L, U, y, yu = train_test_split(X,Y, train_size=0.05, test_size=0.95, stratify=Y)

dsl = DeepSelfLabeling(k=np.size(np.unique(Y)), dim=np.size(L, axis=1),lote=30)

PL, PU = dsl.self_Labeled(L, U, y)

Rot, NaoRot = dsl.divisao_grupos(PU, PL)

centroides = np.array([[1,2,1], [4, 5, 5], [10, 20, 12], [1.1, 2.3, 0.8]])
centros = sca.fit_transform(centroides)
x = centros[np.size(centros, axis=0)-1,:]
centros = centros[0:np.size(centros,axis=0)-1,:]

def tstudent(x, centroides):    
    den = []
    for u in centroides:
        q = 1 / ((1 + (u - x)**2))
        den.append(q)
        
    den = np.array(den)
    soma = np.sum(den, axis=1)
    p = den[0,:] / soma
    return p


den = tstudent(x, centros)
PU['grupo'] = yu
G1 = PU[PU['grupo'] == 1]
G1 = G1.drop(['grupo'], axis=1)
G2 = dados[dados['classe'] == 2]
G2 = G2.drop(['classe'], axis=1)
G3 = dados[dados['classe'] == 3]
G3 = G3.drop(['classe'], axis=1)

"""
G1 = pd.DataFrame(np.array([
        [1,10,0],
        [2,10,0],
        [2,30,0 ],]), columns=['cor','tamanho', 'textura'])
"""
G2 = pd.DataFrame(np.array([
        [1.1,1.2,1.3],
        [1,1,1],
        [0.9,1.1,0.91]]), columns=['cor','tamanho', 'textura'])
"""
G3 = pd.DataFrame(np.array([
        [4,30,1],
        [4,20,1]]), columns=['cor','tamanho', 'textura'])
"""
def entropia(X):
    T = np.size(X, axis=1)
    colunas = X.columns.values
    H = []
    for c in colunas:
        dados = X[c]
        p = dados.value_counts()
        p = p.values / T
        p = -(p * np.log2(p))
        H.append(p.sum())
        
    return np.sum(H)

