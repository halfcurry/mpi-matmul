# Benchmarking Report: Distributed Matrix Multiplication

## 1. Objective

To evaluate serial vs. MPI-based distributed matrix multiplication performance in pure Python, focusing on speedup and efficiency on a 4-core Codespace.

## 2. Experimental Setup

* **Environment**: GitHub Codespaces (4-core CPU, 16GB RAM).
* **Implementation**: Pure Python for matrix operations; `mpi4py` for distribution (row-wise).
* **Matrix Sizes (N)**: 50, 100, 200, 400.
* **Runs per Test**: 3.
* **Processes (MPI)**: 1, 2, 3, 4.

## 3. Results Summary

| N   | Mode            | Avg Time (s) | Speedup | Efficiency |
| :-- | :-------------- | :----------- | :------ | :--------- |
| **50** | Serial          | 0.019440     | 1.00x   | N/A        |
|     | MPI 1 Processes | 0.019572     | 0.99x   | 99.32%     |
|     | MPI 2 Processes | 0.009796     | 1.98x   | 99.22%     |
|     | MPI 3 Processes | 0.012328     | 1.58x   | 52.56%     |
|     | MPI 4 Processes | 0.009249     | 2.10x   | 52.55%     |
| **100**| Serial          | 0.144769     | 1.00x   | N/A        |
|     | MPI 1 Processes | 0.157976     | 0.92x   | 91.64%     |
|     | MPI 2 Processes | 0.083361     | 1.74x   | 86.83%     |
|     | MPI 3 Processes | 0.088502     | 1.64x   | 54.53%     |
|     | MPI 4 Processes | 0.074399     | 1.95x   | 48.65%     |
| **200**| Serial          | 1.184336     | 1.00x   | N/A        |
|     | MPI 1 Processes | 1.170380     | 1.01x   | 101.19%    |
|     | MPI 2 Processes | 0.616879     | 1.92x   | 95.99%     |
|     | MPI 3 Processes | 0.683912     | 1.73x   | 57.72%     |
|     | MPI 4 Processes | 0.569480     | 2.08x   | 51.99%     |
| **400**| Serial          | 9.786854     | 1.00x   | N/A        |
|     | MPI 1 Processes | 9.823064     | 1.00x   | 99.63%     |
|     | MPI 2 Processes | 4.908274     | 1.99x   | 99.70%     |
|     | MPI 3 Processes | 5.422227     | 1.80x   | 60.17%     |
|     | MPI 4 Processes | 4.493096     | 2.18x   | 54.45%     |

## 4. Analysis

* **MPI Overhead**: MPI 1-process times are comparable to or slightly higher than serial due to MPI setup overhead.
* **Near-Linear Speedup (2 Processes)**: Strong speedup (approx. 2x) and high efficiency (approx. 90-100%) are observed with 2 processes, indicating effective parallelization.
* **Diminished Returns (3 & 4 Processes)**: Speedup gains beyond 2 processes are less pronounced (e.g., ~2.1x for 4 processes). Efficiency drops to around 50-60%.
* **Factors**: This diminishing return is attributed to contention for CPU resources (on a 4-core machine), Python's GIL effects (though less direct with separate processes), and communication overhead becoming more significant relative to computation as more processes share limited resources.