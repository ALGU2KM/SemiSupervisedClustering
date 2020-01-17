# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:06:17 2020

@author: brunn
"""
import keras.backend as K
from sklearn.cluster import KMeans
from keras.optimizers import SGD
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from keras.engine.topology import Layer, InputSpec

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
        
        input_img = Input((dim,))
        encoded = Dense(250, activation='relu')(input_img)
        drop = Dropout(0.2)(encoded)
        encoded = Dense(200, activation='relu')(drop)
        drop = Dropout(0.2)(encoded)
        encoded = Dense(100, activation='relu')(drop)
        
        Z = Dense(k, activation='relu')(encoded)
        
        decoded = Dense(100, activation='relu')(Z)
        drop = Dropout(0.2)(decoded)
        decoded = Dense(200, activation='relu')(drop)
        drop = Dropout(0.2)(decoded)
        decoded = Dense(250, activation='relu')(drop)
        decoded = Dense(dim, activation='sigmoid')(decoded)
        
        self.encoder = Model(input_img, Z)
        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(loss='mse', optimizer=SGD(lr=0.1, decay=0, momentum=0.9))
    
    def inicializacao(self, X, epocas = 500):
        self.autoencoder.fit(X, X, epochs=epocas, batch_size=self.lote)
                
        self.kmeans = KMeans(n_clusters=self.k)
        self.y_pred = self.kmeans.fit_predict(self.encoder.predict(X))
        self.centroides = self.kmeans.cluster_centers_
        
        self.DEC = Sequential([self.encoder, ClusteringLayer(self.k, weights=self.centroides, name='Agrupamento')])
        self.DEC.compile(loss='kullback_leibler_divergence', optimizer='adadelta')
        self.DEC.summary()
        
    def p_mat(self, q):
        weight = q**2 / q.sum(0)
        return (weight.T / weight.sum(1)).T
        
class ClusteringLayer(Layer):
    
    def __init__(self, output_dim, input_dim=None, weights=None, alpha=1.0, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.alpha = alpha
        # kmeans cluster centre locations
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(ClusteringLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = K.variable(self.initial_weights)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        q = 1.0/(1.0 + K.sqrt(K.sum(K.square(K.expand_dims(x, 1) - self.W), axis=2))**2 /self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = K.transpose(K.transpose(q)/K.sum(q, axis=1))
        return q

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'input_dim': self.input_dim}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
