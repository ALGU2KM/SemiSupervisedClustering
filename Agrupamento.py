import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import entropy

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

L = pd.read_csv('PL.csv')
U = pd.read_csv('PU.csv')
classes = L['classe']
y = L['grupo']
yu = U['grupo']
L = L.drop(['classe', 'grupo'], axis=1)
U = U.drop(['grupo'], axis=1)

T = pd.concat([U, L])
Tt = tsne.fit_transform(T)
Y = pd.concat([yu, y]).values
t = np.size(y) + np.size(yu)

cores = ['red', 'blue', 'green', 'orange', 'yellow', 'pink']
cinza = ['#1C1C1C', '#D3D3D3', '#363636', '#A9A9A9', '#4F4F4F', '#808080']
marcador = ['*', 's', 'v', 'X', 'P', 'D']
for i, x in enumerate(Tt):
    plt.scatter(x[0], x[1], c = cores[Y[i]], s = 30, marker=marcador[Y[i]])  
    """
    if i < np.size(yu):
        plt.scatter(x[0], x[1], c = cinza[Y[i]], s =20, marker=marcador[Y[i]])
    else:
        plt.scatter(x[0], x[1], c = cores[Y[i]], s = 30, marker=marcador[Y[i]])
   """
   
plt.legend()
plt.show()

p0 = L0[1,:]
p1 = L1[1,:]
p2 = L2[1,:]

q0 = L0[5,:]
q1 = L1[5,:]
q2 = L2[5,:]
