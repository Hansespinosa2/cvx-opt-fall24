import numpy as np
from newtons_method import newton_method_fs_lp

def phase_1_init(A, x_0, b, c,  mu=20, alpha=0.1, beta=0.3, epsilon_inner=1e-8, epsilon_outer=1e-4, max_iter=100):
    # Problem Restructure
    n = x_0.shape[0]
    A_aug = np.c_[A, -A @ np.ones(n)]
    b_aug = b - A @ np.ones(n)
    t_0 =  2 - np.min(x_0)
    c_aug = np.r_[np.zeros(n),1]
    z = np.r_[x_0, t_0]

    results = newton_method_fs_lp(A_aug, z, b_aug, c_aug)

    x_star = results["x"][-1][:-1]
    t_star = results["x"][-1][-1]

    if t_star > 1:
        print(f"""
        Problem is infeasible with t_star = {t_star:.2f}
        """)
        return None
    else:
        print(f"""
    Problem is strictly feasible with t_star = {t_star:.2f}
        """)

    return x_star