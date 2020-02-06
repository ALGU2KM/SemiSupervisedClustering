# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:06:17 2020

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
        
        """
        input_img = Input((dim,))
        encoded = Dense(250, activation='relu')(input_img)
        drop = Dropout(0.2)(encoded)
        encoded = Dense(150, activation='relu')(drop)
        drop = Dropout(0.2)(encoded)
        encoded = Dense(100, activation='relu')(drop)
        
        Z = Dense(k, activation='relu')(encoded)
        
        decoded = Dense(100, activation='relu')(Z)
        drop = Dropout(0.2)(decoded)
        decoded = Dense(150, activation='relu')(drop)
        drop = Dropout(0.2)(decoded)
        decoded = Dense(250, activation='relu')(drop)
        decoded = Dense(dim, activation='sigmoid')(decoded)
                        
        self.encoder = Model(input_img, Z)
        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(loss='mse', optimizer=SGD(lr=0.1, decay=0, momentum=0.9))
                
        #self.autoencoder = load_model('DAE-MNIST64.h5')
        #self.encoder = load_model('encoder.h5')

        """
    
    def inicializacao(self, X, epocas = 500):
        #self.autoencoder.fit(X, X, epochs=epocas, batch_size=self.lote)
        self.dec.initialize(X)
                
        self.kmeans = KMeans(n_clusters=self.k)
        self.y_pred = self.kmeans.fit_predict(self.encoder.predict(X))
        self.centroides = self.kmeans.cluster_centers_
                
        self.DEC = Sequential([self.encoder, ClusteringLayer(self.k, weights=self.centroides, name='Agrupamento')])
        self.DEC.compile(loss='kullback_leibler_divergence', optimizer='adadelta')
        self.DEC.summary()
        
    def p_mat(self, q):
        weight = q**2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

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
        
    def cluster(self, X, y=None,
                tol=0.01, update_interval=None,
                iter_max=1e4,
                save_interval=None,
                **kwargs):

        if update_interval is None:
            # 1 epochs
            update_interval = X.shape[0]/self.batch_size
        #print('Update interval', update_interval)

        if save_interval is None:
            # 50 epochs
            save_interval = X.shape[0]/self.batch_size*50
        #print('Save interval', save_interval)

        assert save_interval >= update_interval

        train = True
        iteration, index = 0, 0
        self.accuracy = []
        print('Clustering...')
        while train:
            
            sys.stdout.write('\r')
            # cutoff iteration
            if iter_max < iteration:
                #print('Reached maximum iteration limit. Stopping training.')
                return self.y_pred

            # update (or initialize) probability distributions and propagate weight changes
            # from DEC model to encoder.
            if iteration % update_interval == 0:
                self.q = self.DEC.predict(X, verbose=0)
                self.p = self.p_mat(self.q)

                y_pred = self.q.argmax(1)
                delta_label = ((y_pred == self.y_pred).sum().astype(np.float32) / y_pred.shape[0])
                if y is not None:
                    acc = self.cluster_acc(y, y_pred)[0]
                    self.accuracy.append(acc)
                    #print('Iteration '+str(iteration)+', Accuracy '+str(np.round(acc, 5)))
                else:
                    a = 2
                    #print(str(np.round(delta_label*100, 5))+'% change in label assignment')

                if delta_label < tol:
                    #print('Reached tolerance threshold. Stopping training.')
                    train = False
                    continue
                else:
                    self.y_pred = y_pred

                for i in range(len(self.encoder.layers)):
                    self.encoder.layers[i].set_weights(self.DEC.layers[0].layers[i].get_weights())
                self.cluster_centres = self.DEC.layers[-1].get_weights()[0]

            # train on batch
            sys.stdout.write('Iteration %d, ' % iteration)
            if (index+1)*self.batch_size > X.shape[0]:
                loss = self.DEC.train_on_batch(X[index*self.batch_size::], self.p[index*self.batch_size::])
                index = 0
                sys.stdout.write('Loss %f |' % loss)
            else:
                loss = self.DEC.train_on_batch(X[index*self.batch_size:(index+1) * self.batch_size],
                                               self.p[index*self.batch_size:(index+1) * self.batch_size])
                sys.stdout.write('Loss %f |' % loss)
                index += 1
            
            # save intermediate
            #if iteration % save_interval == 0:
                #z = self.encoder.predict(X)
                #pca = PCA(n_components=2).fit(z)
                #z_2d = pca.transform(z)
                #clust_2d = pca.transform(self.cluster_centres)
                # save states for visualization
                #pickle.dump({'z_2d': z_2d, 'clust_2d': clust_2d, 'q': self.q, 'p': self.p},
                #            open('c'+str(iteration)+'.pkl', 'wb'))
                # save DEC model checkpoints
                #self.DEC.save('DEC_model_'+str(iteration)+'.h5')

            iteration += 1
            sys.stdout.flush()
        return
        
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
    
