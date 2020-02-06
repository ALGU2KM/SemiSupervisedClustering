import pandas as pd
import numpy as np
import keras.backend as K
from numpy import linalg as LA
from sklearn.mixture import GaussianMixture

x = np.array(
        [
            [11.0,11.0,14.5,12.0,11.0],
            [2.0,3.0,1.0,2.0,2.0]
        ],
        dtype=np.float32
        )

pesos = np.array(
        [
                [2.0,5.0,3.0,2.0,3.0],
                [6.0,3.0,5.0,4.0,7.0],
                [10.0,12.0,15.0,14.0,13.0],
                [0.1,0.2,0.3,0.1,0.4]
                ], 
        dtype=np.float32
        )

def parzen_window_est(x_samples, h=1, center=[0,0,0]):
    '''
    Implementation of the Parzen-window estimation for hypercubes.

    Keyword arguments:
        x_samples: A 'n x d'-dimensional numpy array, where each sample
            is stored in a separate row.
        h: The length of the hypercube.
        center: The coordinate center of the hypercube

    Returns the probability density for observing k samples inside the hypercube.

    '''
    dimensions = x_samples.shape[1]

    #assert (len(center) == dimensions),  
    'Number of center coordinates have to match sample dimensions'
    k = 0
    for x in x_samples:
        is_inside = 1
        for axis,center_point in zip(x, center):
            if np.abs(axis-center_point) > (h/2):
                is_inside = 0
        k += is_inside
    return (k / len(x_samples)) / (h**dimensions)

print('p(x) =', parzen_window_est(x[0,:], h=1, center=pesos[2,:]))

def student(x, c):
    v = []
    for w in pesos:
        norma = LA.norm(x - w)
        den = 1 / np.sqrt((1 + norma**2))
        v.append(den)
    den = np.sum(v)
    q = v / den
    return q

d = student(x, 0)
