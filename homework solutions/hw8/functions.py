import numpy as np
def f(x:np.array, c:np.array):
    x = np.maximum(x, 1e-8)
    return c.T @ x - np.sum(np.log(x))

def g(x:np.array, c:np.array):
    return c - 1/x

def H(x:np.array,c:np.array = None):
    H = np.diag((x**(-2)))
    eI = np.identity(x.shape[0]) * 10e-9
    return H + eI

def H_inv(x: np.array, c:np.array = None):
    H_inv = np.diag((x**(2)))
    eI = np.identity(x.shape[0]) * 10e-9
    return H_inv + eI