import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('./notears')
from notears.nonlinear import NotearsMLP, notears_nonlinear
from notears.utils import (
    set_random_seed,
    simulate_dag,
    simulate_nonlinear_sem,
    count_accuracy
)
from CausalDisco.analytics import var_sortability

def run_nonlinear_notears_experiment(
    d=5,
    s0=9,
    graph_type="ER",
    sem_type="mlp",
    n=1000,
    noise_scale=1.0,
    standardize=False,
    lambda1=0.01,
    lambda2=0.01,
    max_iter=100,
    random_seed=42
):
    """
    1) Simulate a DAG with `simulate_dag`.
    2) Generate data with `simulate_nonlinear_sem` (sem_type can be 'mlp','gp','mim', etc.).
    3) Possibly standardize.
    4) Fit NotearsMLP with `notears_nonlinear`.
    5) Evaluate accuracy (including SHD).
    """
    set_random_seed(random_seed)

    # 1. DAG adjacency
    B_bin = simulate_dag(d, s0, graph_type)

    # 2. Nonlinear data
    X = simulate_nonlinear_sem(B_bin, n, sem_type)

    # 3. Standardize if requested
    if standardize:
        X = StandardScaler().fit_transform(X)
    X = X.astype(np.float32)

    # 4. Run nonlinear NOTEARS
    model = NotearsMLP(dims=[d, 10, 1], bias=True)
    W_est = notears_nonlinear(
        model, 
        X, 
        lambda1=lambda1, 
        lambda2=lambda2,
        max_iter=max_iter,
        h_tol=1e-8,
        rho_max=1e16,
        w_threshold=0.3
    )

    # 5. Evaluate
    W_est_bin = (np.abs(W_est) > 1e-8).astype(int)
    acc = count_accuracy(B_bin, W_est_bin)
    
    vs_score = var_sortability(X, B_bin)

    return {
        "fdr": acc["fdr"],
        "tpr": acc["tpr"],
        "fpr": acc["fpr"],
        "shd": acc["shd"],
        "nnz": acc["nnz"],
        "varsortability": vs_score,
        "W_est": W_est
    }

def hyperparam_sweep_nonlinear():
    """
    Example function to show how we might test the impact of lambda1/lambda2
    on the nonlinear NOTEARS approach.
    """
    lambda1_values = [0.0, 0.001, 0.01, 0.1]
    results = []

    for lam1 in lambda1_values:
        r = run_nonlinear_notears_experiment(
            d=5, 
            s0=5, 
            graph_type="ER",
            sem_type="mlp", 
            n=1000,
            noise_scale=1.0,
            standardize=False,
            lambda1=lam1,
            lambda2=0.01,
            max_iter=100,
            random_seed=123
        )
        r["lambda1"] = lam1
        results.append(r)

    # Plot example: TPR vs. lambda1
    plt.figure(figsize=(6,4))
    xvals = [r["lambda1"] for r in results]
    tpr_vals = [r["tpr"] for r in results]
    plt.plot(xvals, tpr_vals, marker='o')
    plt.title("Effect of lambda1 on TPR (Nonlinear NOTEARS)")
    plt.xlabel("lambda1")
    plt.ylabel("TPR (unitless)")
    plt.xscale('log')
    plt.grid(True)
    plt.show()

    return results

def main():
    # Example usage
    results = run_nonlinear_notears_experiment(
        d=5, 
        s0=5, 
        graph_type="ER", 
        sem_type="mlp",
        n=1000, 
        noise_scale=1.0,
        standardize=False, 
        lambda1=0.01, 
        lambda2=0.01,
        max_iter=100,
        random_seed=123
    )
    print("Nonlinear NOTEARS results:", results)

    metrics = ["fdr", "tpr", "fpr", "shd", "varsortability"]
    print({m: results[m] for m in metrics})
    _ = hyperparam_sweep_nonlinear()

if __name__ == "__main__":
    main()
