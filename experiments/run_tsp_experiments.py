import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.simulated_annealing import simulated_annealing
from src.genetic_algorithm import genetic_algorithm
from src.mwi_h import mwi_h

def main():
    np.random.seed(42)

    # Load distance matrix
    dist_df = pd.read_excel(
        "data/TSP_50_Cities_Data.xlsx",
        sheet_name="Distance_Matrix",
        index_col=0
    )
    D = dist_df.values

    iterations = 800

    sa_hist = simulated_annealing(D, iterations=iterations)
    ga_hist = genetic_algorithm(D, iterations=iterations)
    mwi_hist = mwi_h(D, iterations=iterations)

    # Plot Figure 3
    plt.figure(figsize=(8, 5))
    plt.plot(sa_hist, label="Simulated Annealing")
    plt.plot(ga_hist, label="Genetic Algorithm")
    plt.plot(mwi_hist, label="MWI-H (proposed)")
    plt.xlabel("Iteration")
    plt.ylabel("Best tour length")
    plt.title("Figure 3: Convergence comparison on 50-city TSP")
    plt.legend()
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out_path = "figures/Figure_3_TSP_Convergence.png"
    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
