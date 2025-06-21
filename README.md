# Distributed Matrix Multiplication using MPI (Python)

## Objective

Implement and evaluate parallel matrix multiplication using MPI in Python. This project compares serial (pure Python) vs. distributed (MPI) matrix multiplication, focusing on data partitioning, communication, and performance benchmarking.

## Notes

* Pure Python matrix operations (no NumPy for core computation).

* MPI-based row-wise data distribution with `mpi4py`.

* Timing and benchmarking: measures execution time, calculates speedup and efficiency.

* Verification against NumPy's `np.dot` (NumPy used *only* for ground truth).

* Configurable matrix dimension ($N$).

* GitHub Codespaces ready with pre-installed `mpi4py`, `numpy`, and `Open MPI` via `devcontainer.json`.

## Initial Implementation

### For serial execution
```bash
python serial_matrix_multiply.py 50
```

### For distributed execution

```bash
mpiexec -n 1 python mpi_matrix_multiply.py 50
mpiexec -n 2 python mpi_matrix_multiply.py 100
mpiexec -n 4 python mpi_matrix_multiply.py 150
```

## Combining Implementations and Benchmarking

  * `matrix_multiplier.py`: Contains serial and MPI matrix multiplication logic.

  * `benchmark_runner.py`: Orchestrates running `matrix_multiplier.py` for performance tests.

## Run Benchmarking

```bash
python benchmark_runner.py <matrix_dimension_N> <num_runs_per_test>
```

**Example**: `python benchmark_runner.py 100 3` (runs with $N=100$, 3 times each mode).

### Direct Testing (`matrix_multiplier.py`)

  * **Serial**: `python matrix_multiplier.py --mode serial -N 50`

  * **MPI**: `mpiexec --oversubscribe -n 4 python matrix_multiplier.py --mode mpi -N 50`

**Note**: Pure Python matrix multiplication is very slow. Start with small $N$ values (e.g., 50-150).

## Performance Analysis

`benchmark_runner.py` outputs a summary table including:

  * **Avg Time (s)**: Average execution time.

  * **Std Dev (s)**: Consistency of runs.

  * **Speedup**: `Time_serial / Time_parallel`. Expect speedup with more processes.

  * **Efficiency**: `Speedup / Number of Processes`. Indicates resource utilization.