import cvxpy as cp
import numpy as np
from newtons_method import newton_method_fs_lp, phase_1_init

def cvx_lp(A, b, c):
    m, n = A.shape

    x = cp.Variable(n)

    constraints = [
        A @ x == b,
        x >= 0
    ]

    obj = cp.Minimize(c.T @ x)

    prob = cp.Problem(obj, constraints)

    prob.solve()
    
    return prob, x


def andy_lp(A, b, c):
    x_0 = np.linalg.lstsq(A, b, rcond=None)[0]

    x_0 = phase_1_init(A,x_0, b, c)

    results = newton_method_fs_lp(A, x_0, b, c)

    return results