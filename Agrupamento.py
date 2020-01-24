import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import entropy
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

L = pd.read_csv('PL.csv')
U = pd.read_csv('PU.csv')

"""
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
    #plt.scatter(x[0], x[1], c = cores[Y[i]], s = 30, marker=marcador[Y[i]])  
    
    if i < np.size(yu):
        plt.scatter(x[0], x[1], c = cinza[Y[i]], s =20, marker=marcador[Y[i]])
    else:
        plt.scatter(x[0], x[1], c = cores[Y[i]], s = 30, marker=marcador[Y[i]])
   
   
plt.legend()
plt.show()
"""

def calcular_classe(probabilidades):
    c = -1
    for i, p in enumerate(probabilidades):
        pr = np.round(p)
        if pr == 1.:
            c = i
            break
    return c

def rotular_amostras(x, L, y, k, t):

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
    classe = calcular_classe(P)
    
    return classe

rotulos = np.zeros(np.size(U, axis=0))-1
estado = rotulos.copy()
k = 6
   
for i in np.arange(k):
   
    GL = L[L['grupo']==i] 
    GU = U[U['grupo']==i]
    indice = GU.index.values
    
    gu = GU['grupo'].values
    gl = GL['grupo'].values
    yl = GL['classe'].values
    yu = GU['classe'].values
    U = GU.drop(['classe','grupo'], axis=1).values
    L = GL.drop(['classe','grupo'], axis=1).values
    
    classes = np.unique(yl)
    
    grupo = np.zeros(np.size(yu))-1
    
    for a, x in enumerate(U):
        print('Amostra: ', a)
        r = rotular_amostras(x, L, yl, 5, 0.1)
        rotulos[indice[a]]  = r
    
    

    
colunas = ['a','b','classe']  

A = pd.DataFrame(np.array(
    [
     [1,1,-1],
     [2,2,1],
     [2,3,2],
     [6,6,-1],
     [7,8,5],
     [6,7,-1],
     [5,6,-1],
     [7,9,2],
     [5,4,1],
     [8,9,-1]
     ]
    ), columns=colunas)   

B = pd.DataFrame(np.array(
    [
     [1,2,1],
     [3,2,3],
     [4,2,1]
     ]
    ),columns=colunas)   
    
    
R = A[A['classe'] != -1]
UR = A[A['classe'] == -1]
A = UR
B = pd.concat([B, R])





















