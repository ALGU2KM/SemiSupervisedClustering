# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import entropy
from sklearn.neighbors import KNeighborsClassifier

class Semi_Supervised_KNN:
    
    def kl_divergence(p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))
    
    def classificar(self, L, U, y, k=3, t = 0.1):
        rotulos = np.zeros(np.size(U, axis=0), dtype=np.int64) - 1
        for i, x in enumerate(U):
            print('Rotulando ', i)
            c = self.rotular_amostras(x, L, y, k, t)
            rotulos[i] = c
        
        """ Rotulando os Remanescentes """
        dados = pd.DataFrame(U)
        dados['y'] = rotulos
        
        rotulados = dados[dados['y'] != -1]
        nrot = dados[dados['y'] == -1]
        indices = nrot.index.values
        
        #Caso alguém não tenho rótulo
        if np.size(indices) != 0:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(rotulados.drop(['y'], axis=1).values, rotulados['y'].values)
            
            for i, x in enumerate(nrot.drop(['y'], axis=1).values):
                c = knn.predict([x])
                pos = indices[i]
                rotulos[pos] = c        
        
        return rotulos
    
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
        vizinhos = vizinhos[vizinhos[dis]<=t]        
        
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