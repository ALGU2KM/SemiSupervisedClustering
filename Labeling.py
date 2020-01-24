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
                 lote=256):
        
        self.dim = dim
        self.k = k
        self.aplha = 1.0
        self.lote = lote
        self.taxaAprendizado = taxaAprendizado
        self.dec = DeepEmbeddingClustering(self.k, self.dim)
        
    def selfLabeled(self, L, U, y):
        """ INICIALIZAÇÃO """
        
        self.dec.initialize(U)
        grupos = self.dec.cluster(U)
        PL = pd.DataFrame(dec.DEC.predict(L))
        PU = pd.DataFrame(dec.DEC.predict(U))
        PL['grupo'] = dec.DEC.predict_classes(L)
        PL['classe'] = y
        PU['grupo'] = dec.DEC.predict_classes(U)
        
        """ ROTULAÇÃO """
        for i in np.arange(self.k):
            
            pass
