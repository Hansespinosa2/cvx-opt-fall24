import numpy as np
from functions import f, g, H, H_inv

def append_results(results, x, f_x, g_x, H_x, dx, lambda_2, iter, nu, gap = None):
    results["x"].append(x)
    results["f_x"].append(f_x)
    results["g_x"].append(g_x)
    results["H_x"].append(H_x)
    results["dx"].append(dx)
    results["lambda_2"].append(lambda_2)
    results["iter"].append(iter)
    results["nu"].append(nu)
    if gap is not None:
        results["gap"].append(gap)

def backtracking_line_search(x, dx, alpha, beta, c):
    t = 1
    while (f(x + t * dx,c) > f(x,c) + alpha * t * g(x,c).T @ dx) or np.isnan(f(x + t * dx,c)):
        t *= beta
        if t < 1e-20:
            break
    return t

def newtons_centering_step(A, x_0, b, c, alpha, beta, epsilon, max_iter):
    # Storing results
    results = {
        "x": [],
        "f_x" : [],
        "g_x" : [],
        "H_x" : [],
        "dx": [],
        "lambda_2": [],
        "iter": [],
        "nu": [],
    }
    
    # Givens
    x = x_0
    i = 0


    while i <= max_iter:
        x = np.maximum(x, 1e-8)
        # Compute gradients and hessians
        f_x = f(x, c)
        g_x = g(x, c)
        H_x = H(x, c)
        H_x_inv = H_inv(x, c)
        

        # Compute Newton step and decrement
        w = np.linalg.solve((A @ H_x_inv @ A.T), (-A @ H_x_inv @ g_x))
        dx_nt = - H_x_inv @ (A.T @ w + g_x)
        nt_dec_2 = - dx_nt.T @ g_x
        
        # Stopping Criterion
        if nt_dec_2 / 2 <= epsilon:
            break

        # Line Search
        t_search = backtracking_line_search(x, dx_nt, alpha, beta, c)
        
        # Update
        x = x + t_search * dx_nt

        # Iteration update
        i += 1

        # Append to results lists
        append_results(results, x, f_x, g_x, H_x, dx_nt, nt_dec_2, i, w)

    # Newton Complete Certificate
    p_star = x.shape[0] + np.sum(np.log(np.maximum(c + A.T @ w,1e-9))) - w.T @ b
    print(f"""
    Newtons Step completed in {i} iterations
    Duality Gap f(x^*) - p^* = {f(x,c) - p_star}
    KKT conditions satisfied with accuracy of {np.linalg.norm(g(x,c) + w.T @ A,2)}
    """)
    return results

def newton_method_fs_lp(A, x_0, b, c, mu=20, alpha=0.1, beta=0.3, epsilon_inner=1e-8, epsilon_outer=1e-4, max_iter=100):
    # Results
    results = {
        "x": [],
        "f_x" : [],
        "g_x" : [],
        "H_x" : [],
        "dx": [],
        "lambda_2": [],
        "iter": [],
        "nu": [],
        "gap": [],
    }

    # Initialize variables
    x = np.maximum(x_0, 1e-8)
    i = 0

    # Initialize t
    t = 1 # DO THIS MORE INTELLIGENTLY

    while i <= max_iter:
        # Centering step
        inner_step_results = newtons_centering_step(A, x, b, c * t, alpha, beta, epsilon_inner, max_iter)
        x_t = inner_step_results["x"][-1]

        # Updating x
        x = x_t

        gap = x.shape[0] / t
        if gap <= epsilon_outer:
            break

        t *= mu

        append_results(results, 
                       x, 
                       inner_step_results["f_x"][-1] / t, 
                       inner_step_results["g_x"][-1], 
                       inner_step_results["H_x"][-1], 
                       inner_step_results["dx"][-1],
                       inner_step_results["lambda_2"][-1],
                       inner_step_results["iter"][-1],
                       inner_step_results["nu"][-1],
                       gap, 
                       )
    
    return results
