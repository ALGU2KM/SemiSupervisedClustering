import pandas as pd
import numpy as np
from numpy import linalg

def student(x, centros, i=0):
    v = []
    for w in centros:
        norma = linalg.norm(x - w)
        den = 1 / np.sqrt((1 + norma**2))
        v.append(den)
    den = np.sum(v)
    q = v / den
    return q

def entropia_cluster(X):
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

def divergencia_KullbackLeibler(p, q):
    d = p * np.log(p / q)
    d = np.sum(d)
    return d