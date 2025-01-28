import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from CausalDisco.analytics import var_sortability

import sys
sys.path.append('./notears')
from notears.utils import (
    set_random_seed,
    simulate_dag,
    simulate_linear_sem,
    count_accuracy
)
from notears.linear import notears_linear

# Define color palette
COLOR_PALETTE = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta']

def run_linear_notears_experiment(
    d=100,
    s0=50,
    graph_type="ER",
    sem_type="gauss",
    n=1000,
    noise_scale=1.0,
    standardize=False,
    lambda1=0.01,
    loss_type="l2",
    w_threshold=0.3,
    random_seed=42
):
    """
    1) Create a random DAG with `simulate_dag`.
    2) Generate linear SEM data with `simulate_linear_sem`.
    3) Optionally standardize data (this can affect the effective 'causal weight' scale).
    4) Run linear NOTEARS.
    5) Evaluate structure accuracy (FDR, TPR, SHD...) + varsortability (optional).
    """
    set_random_seed(random_seed)

    # 1. Generate random DAG adjacency (B_bin) with expected #edges = s0
    B_bin = simulate_dag(d, s0, graph_type)

    # 2. Generate linear SEM data from that adjacency
    X = simulate_linear_sem(B_bin, n, sem_type, noise_scale)

    # 3. Standardize if requested
    if standardize:
        X = StandardScaler().fit_transform(X)
    
    # 4. Run linear NOTEARS
    W_est = notears_linear(
        X, 
        lambda1=lambda1, 
        loss_type=loss_type, 
        w_threshold=w_threshold
    )

    # 5. Evaluate structure accuracy
    W_est_bin = (np.abs(W_est) > 1e-8).astype(int)
    acc = count_accuracy(B_bin, W_est_bin)

    # varsortability
    try:
        vs_score = var_sortability(X, B_bin)
    except:
        vs_score = None

    return {
        "fdr": acc["fdr"],
        "tpr": acc["tpr"],
        "fpr": acc["fpr"],
        "shd": acc["shd"],
        "nnz": acc["nnz"],
        "varsortability": vs_score,
        "W_est": W_est
    }

def hyperparam_sweep_linear():
    """
    Assess the impact of the lambda1 hyperparameter on linear NOTEARS performance.
    """
    lambda1_values = np.logspace(-4, 1, 10)
    results = []

    for lam in lambda1_values:
        r = run_linear_notears_experiment(
            d=5, s0=5, graph_type="ER",
            sem_type="gauss", n=1000, noise_scale=1.0,
            standardize=False,
            lambda1=lam,
            loss_type="l2",
            w_threshold=0.3,
            random_seed=123
        )
        r["lambda1"] = lam
        results.append(r)

    # Plot results (SHD vs. lambda1)
    plt.figure(figsize=(8, 5))
    xvals = [r["lambda1"] for r in results]
    shd_vals = [r["shd"] for r in results]
    plt.plot(xvals, shd_vals, marker='o', color="#AEC6CF", label="SHD")
    plt.xscale('log')
    plt.xlabel("lambda1 (log scale)", fontsize=12)
    plt.ylabel("SHD (Structural Hamming Distance)", fontsize=12)
    plt.title("Effect of lambda1 on SHD (Linear NOTEARS)", fontsize=14)
    plt.grid(True, which="both", linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.show()

    return results

def main():
    d = 20
    s0 = 10
    # Compare raw vs. standardized data
    results_raw = run_linear_notears_experiment(
        d=d, s0=s0, graph_type="ER", sem_type="gauss",
        n=100, noise_scale=1.0,
        standardize=False,
        lambda1=0.01, loss_type="l2",
        random_seed=123
    )
    results_std = run_linear_notears_experiment(
        d=d, s0=s0, graph_type="ER", sem_type="gauss",
        n=100, noise_scale=1.0,
        standardize=True,
        lambda1=0.01, loss_type="l2",
        random_seed=123
    )

    print("Linear NOTEARS (Raw data):", results_raw)
    print("Linear NOTEARS (Standardized data):", results_std)

    # Define metrics
    metrics_small = ["fdr", "tpr", "fpr", "varsortability"]
    metric_large = ["shd"]

    # Replace None with 0 or another placeholder and scale metrics_small to 0-100
    raw_small_vals = [results_raw[m] * 100 if results_raw[m] is not None else 0 for m in metrics_small]
    std_small_vals = [results_std[m] * 100 if results_std[m] is not None else 0 for m in metrics_small]
    raw_large_vals = [results_raw[m] if results_raw[m] is not None else 0 for m in metric_large]
    std_large_vals = [results_std[m] if results_std[m] is not None else 0 for m in metric_large]

    # Define x positions
    x_small = np.arange(len(metrics_small))
    x_large = np.arange(len(metric_large))

    width = 0.35  # Width of the bars

    # Create a figure with two subplots side by side with 70% and 30% widths
    fig, (ax1, ax2) = plt.subplots(
        1, 2, 
        figsize=(16, 6), 
        gridspec_kw={'width_ratios': [7, 3]}
    )

    # Plot 1: Metrics
    ax1.bar(x_small - width/2, raw_small_vals, width, label="Raw", color=COLOR_PALETTE[0])
    ax1.bar(x_small + width/2, std_small_vals, width, label="Standardized", color=COLOR_PALETTE[1])
    ax1.set_xticks(x_small)
    ax1.set_xticklabels(metrics_small, ha='center')
    ax1.set_ylim(0, 100)
    ax1.set_title("Linear NOTEARS: Raw vs. Standardized", fontsize=14)
    ax1.set_xlabel("Metrics", fontsize=12)
    ax1.set_ylabel("Percentage", fontsize=12)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Plot 2: SHD
    ax2.bar(x_large - width/2, raw_large_vals, width, label="Raw", color=COLOR_PALETTE[0])
    ax2.bar(x_large + width/2, std_large_vals, width, label="Standardized", color=COLOR_PALETTE[1])
    ax2.set_xticks(x_large)
    ax2.set_xticklabels(metric_large, ha='center')
    
    ax2.set_title("Linear NOTEARS: Raw vs. Standardized", fontsize=14)
    ax2.set_ylabel("SHD", fontsize=12)
    ax2.legend(fontsize=12, loc='upper right')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
    _ = hyperparam_sweep_linear()


if __name__ == "__main__":
    main()
