import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as sc
from scipy.stats import norm

MinMax = MinMaxScaler()

dados = pd.read_csv('d:/basedados/iris.csv')
X = MinMax.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values

def KL(P,Q):
     epsilon = 0.00001
    
     # You may want to instead make copies to avoid changing the np arrays.
     P = P+epsilon
     Q = Q+epsilon
     
     divergence = np.sum(P*np.log(P/Q))
     return divergence

resultado = []

p = np.array([0.1,0.1,0.8])

q = np.array([0.4,0.4,0.2])
r = np.array([0.9,0.1,0.1])

resultado.append(KL(p,q))
resultado.append(KL(p, r))

entr = []
entr.append(entropia(0.1))
entr.append(entropia(0.9))


def entropia(v):
    return v*np.log2(v)


def entropia_total(distribuicao):
    res = []
    for d in distribuicao:
        res.append(entropia(d))
    res = np.array(res)
    som = res.sum(axis=0)
    return -som


print(sc.entropy(X[0,:], X[10,:]))