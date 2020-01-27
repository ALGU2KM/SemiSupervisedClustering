import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from keras.utils import np_utils
from Labeling import DeepSelfLabeling
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import accuracy_score

k = 6


sca = MinMaxScaler()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

dados = pd.read_csv('d:/basedados/agricultura.csv')
dados = dados[dados['classe'] < k]
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values

L, U, y, yu = train_test_split(X,Y, train_size=0.05, test_size=0.95, stratify=Y)

dsl = DeepSelfLabeling(k=5, dim=np.size(L, axis=1),lote=120)

PL, PU = dsl.self_Labeled(L, U, y)

Rot, NaoRot = dsl.divisao_grupos(PU, PL)
