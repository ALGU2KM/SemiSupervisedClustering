import pandas as pd
import numpy as np
from Clustering import SemiClustering
from sklearn.preprocessing import MinMaxScaler

sca = MinMaxScaler()

dados = pd.read_csv('d:/basedados/mnist.csv')
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values


sdc = SemiClustering(10, np.size(X, 1))
sdc.treinar(X, 100)