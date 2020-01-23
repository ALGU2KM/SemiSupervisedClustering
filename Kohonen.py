import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from keras.utils import np_utils
from DEC import DeepEmbeddingClustering
from sklearn.model_selection import train_test_split
from keras.models import load_model

k = 3

sca = MinMaxScaler()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

dados = pd.read_csv('d:/basedados/mnist64.csv')
dados = dados[dados['classe'] < k]
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values

from minisom import MiniSom    
som = MiniSom(10, 10, np.size(X, axis=1), sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
som.train(X, 500) # trains the SOM with 100 iterations

som.winner(X)