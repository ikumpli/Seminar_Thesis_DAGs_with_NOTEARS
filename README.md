# Seminar Thesis on NOTEARS Algorithm

This repository contains the experiments, code, and results for my seminar thesis on the **NOTEARS Algorithm**, a method for learning Directed Acyclic Graphs (DAGs) from observational data. The repository is structured to include both linear and nonlinear versions of the NOTEARS algorithm, along with the corresponding plots and experiment notebooks.

## Key Components

1. **notears/ Directory**:
   - Contains the original implementation of the NOTEARS algorithm by the authors.
   - This ensures we have access to their exact methods for comparison and reuse in experiments.

2. **Experiment Notebooks**:
   - `Exp_1_Varsortability.ipynb`: Explores the impact of **varsortability** on the performance of NOTEARS using simulated DAGs.
   - `Exp_2_3_Performance_comparison.ipynb`: Contains experiments comparing **Linear vs. Nonlinear NOTEARS** (Experiment 2) and assessing the impact of hyperparameters (Experiment 3).

3. **Code Files**:
   - `notears_linear.py`: Contains functions for performing experiments using the **linear NOTEARS** method.
   - `notears_nonlinear.py`: Contains functions for performing experiments using the **nonlinear NOTEARS** method (e.g., MLP-based SEMs).

4. **Plots**:
   - All experiment results are visualized as plots saved in the `plots/` directory. These include:
     - Experiment 1: Varsortability plots.
     - Experiment 2: Comparison of Linear vs. Nonlinear NOTEARS.
     - Experiment 3: Hyperparameter assessment results.

5. **Dependencies**:
   - A `requirements.txt` file is provided, listing all the Python packages needed to run the experiments.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Conda or virtualenv (recommended for dependency management)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/notears-experiments.git
   cd notears-experiments


## Acknowledgments
- The original implementation of the NOTEARS algorithm is credited to the authors of the paper: **Zheng et al., "DAGs with NO TEARS: Continuous Optimization for Structure Learning"**. You can find the original code here: [Original NOTEARS Repository](https://github.com/xunzheng/notears).
