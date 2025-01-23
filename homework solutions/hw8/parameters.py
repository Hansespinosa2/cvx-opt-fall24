import numpy as np
def create_problem_params(m=100,n=500, seed = None):
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randn(m-1,n) * 10
    A = np.r_[A,np.random.rand(n).reshape(1,-1)]
    b = np.random.rand(m)
    x_0 = np.linalg.lstsq(A, b, rcond=None)[0]
    c = np.random.rand(n) * 20 + 5

    return A, x_0, b, c