from DEC import DeepEmbeddingClustering
import numpy as np
import pandas as pd

class DeepSelfLabeled:
    
    def __init__(self, k, t):
        self.k = k
        self.t = t
    
    def inicializacao(self, L, U, y):
        
        #INICIALIZAÇÃO 
        self.dec = DeepEmbeddingClustering(self.k, np.size(U, axis=1))
        self.dec.initialize(U)
        self.dec.cluster(U)
        self.G = pd.DataFrame(U)
        self.G['g'] = self.dec.predict_classes(U)

    def rotulacao(L, U, y):    
        self.Lz = self.dec.DEC.encoder.predict(L)       #Calcula L no espaço Z
        self.Lg = self.dec.predict(L)                   #Calcula os grupos para L
        GL = pd.DataFrame(GL)
        GL['g'] = self.dec.predict_classes(L)
        GL['y'] = y
        
        for i in np.arange(self.k):
            self.trabalhar_grupo(self.G[self.G['g']==i], GL[GL['g']==i])
     
    def trabalhar_grupo(self, G, GL):
        GL = GL.drop(['g'], axis=1)         
        G = G.drop(['g'], axis=1)
        
        pg = self.dec.predict(G.values)                  #Calcula distribuição de probabilidade dos elementos do grupo
        centroides,self.classes = criar_centroides(G)    #Calcula os centroides de cada classe
        centros = self.dec.predict(centroides)           #calcula distribuição de probabilidade de cada centro
        
        #CALCULAR DIVERGÊNCIA DOS ELEMENTOS PARA CADA GRUPOS
        rotulos = []
        for i, p in enumerate(pg):
            r = self.calcular_rotulo(p, centros)        
        
                
    def criar_centroides(self, L):
        classes = L['y'].unique()
        #L = L.drop(['g'], axis=1)
        centroides = L.groupby(['y']).mean()
        return (centroides.values, classes)
    
    def calcular_rotulo(self,p, centros):
        rotulo = self.classes[0]
        primeiro = centros[0,:]
        menorD = KL(p, centros[0,:]) 
        for i, q in enumerate(centros):
            d = KL(p, q)
            if d < menorD:
                menorD = d
                rotulo = self.classes[i]
        return menor
    
    def KL(P,Q):
        epsilon = 0.00001
        P = P+epsilon
        Q = Q+epsilon
     
        divergence = np.sum(P*np.log(P/Q))
     return divergence    
        
        
             
            