import theano
import numpy as np

def init_W(shape):
    std = 2.*(6./(shape[0] + shape[1]))**(0.5)
    M = np.random.standard_normal(shape).astype(dtype=theano.config.floatX)
    if shape[0] == shape[1]:
        u, s, v = np.linalg.svd(M)
        M = np.dot(u, v)
    else:
        M = M*std
    M = M*std
    return theano.shared(M)

def init_b(shape):
    return theano.shared(np.zeros(shape).astype(dtype=theano.config.floatX))
