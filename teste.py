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