# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 21:07:16 2020

@author: brunn
"""

import keras.backend as K
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from keras.optimizers import SGD
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, Input
from keras.engine.topology import Layer, InputSpec
from DEC import DeepEmbeddingClustering
from scipy.stats import entropy

class DeepSelfLabeling:
    
    def __init__(self,
                 k=10,
                 dim=784,
                 taxaAprendizado = 0.1, 
                 t=0.1,
                 lote=256):
        
        self.dim = dim
        self.k = k
        self.t = t
        self.aplha = 1.0
        self.lote = lote
        self.taxaAprendizado = taxaAprendizado
        self.dec = DeepEmbeddingClustering(self.k, self.dim, batch_size=self.lote)
        
    def self_Labeled(self, L, U, y):
        PL, PU = self.inicializacao(L, U, y)
        return PL, PU
    
    """ INICIALIZAÇÃO DO MODELO E DO AGRUPAMENTO """
    def inicializacao(self, L, U, y):
        """ INICIALIZAÇÃO """
        
        indiceL = np.arange(np.size(L, axis=0))
        indiceU = np.arange(np.size(L, axis=0), np.size(L, axis=0) + np.size(U, axis=0))
        
        self.rotulos = np.zeros(np.size(U, axis=0))-1
        self.dec.initialize(U)
        grupos = self.dec.cluster(U)
        PL = pd.DataFrame(self.dec.DEC.predict(L), index=indiceL)
        PL['classe'] = y
        PL['grupo'] = self.dec.DEC.predict_classes(L)
        PU = pd.DataFrame(self.dec.DEC.predict(U), index=indiceU)
        PU['grupo'] = self.dec.DEC.predict_classes(U)
        self.fi = np.size(L, axis=0)
        
        return PL, PU
    
    
    """ROTULAÇÇÃO DE ELEMENTOS E DIVISÃO DOS GRUPOS """
    def divisao_grupos(self, U, L):
        y = L['classe'].values
        gl = L['grupo'].values
        indiceL = L.index.values
        L = L.drop(['grupo'], axis=1)
        
        """ DIVISÃO DOS GRUPOS """
        indice = U.index.values

        for i in np.arange(self.k):
            Ut = U[U['grupo'] == i]
            Ut = Ut.drop(['grupo'], axis=1).values
            
            for a, x in enumerate(Ut):
                r = self.rotular_amostras(x, L.drop(['classe'], axis=1).values, y, self.k, self.t)
                self.rotulos[indice[a]-self.fi]  = r

        """ Remoção dos elementos rotulados """
        Ut = U.drop(['grupo'], axis=1)
        Ut['classe'] = self.rotulos
        novos = Ut[Ut['classe'] != -1]
        L = pd.concat([L, novos])
        Ut = Ut[Ut['classe']==-1]
        Ut = Ut.drop(['classe'], axis=1)
        return L, Ut
        
    def calcular_centroides(self, X):
        pass
            
        
    def rotular_amostras(self, x, L, y, k, t):

        """ Calculando distância da Amostra para cada elemento de L """        
        dis = []
        for xr in L:
            #dis.append(distance.euclidean(x, xr))
            divergencia = entropy(x, xr)
            dis.append(divergencia)
        
        """ Descobrindo os k vizinhos rotulados menos divergentes """
        rot = pd.DataFrame(L)
        rot['y'] = y
        
        
        rot['dis'] = dis
        rot = rot.sort_values(by='dis')
        vizinhos = rot.iloc[0:k,:]
        vizinhos = vizinhos[vizinhos['dis']<=t]        
        
        """ Caso não existem vizinhos rotulados suficientes """
        if np.size(vizinhos, axis=1) < k:
            return -1
        
        """ Calculando as Classes """
        classes = np.unique(y)
        P = []
        for c in classes:
            q = (vizinhos['y'] == c).sum()
            p = q / k
            P.append(p)
        classe = self.calcular_classe(P)
        
        return classe
    
    def calcular_classe(self, probabilidades):
        c = -1
        for i, p in enumerate(probabilidades):
            pr = np.round(p)
            if pr == 1.:
                c = i
                break
        return c