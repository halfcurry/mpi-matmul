# matrix_multiplier.py
import time
import random
import sys
import numpy as np # Only for verification

# Conditional import for MPI
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

# --- Helper Functions ---
def create_matrix(rows, cols):
    """Creates a matrix (list of lists) filled with random floats."""
    matrix = []
    for _ in range(rows):
        row = [random.random() for _ in range(cols)]
        matrix.append(row)
    return matrix

def serial_matrix_multiply(A, B):
    """
    Performs standard matrix multiplication C = A * B using nested loops.
    A is rows x common_dim
    B is common_dim x cols
    C is rows x cols
    """
    rows_A = len(A)
    common_dim = len(A[0]) # Number of columns in A, which is rows in B
    cols_B = len(B[0])

    if common_dim != len(B):
        raise ValueError("Matrix A columns must match Matrix B rows for multiplication.")

    C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(common_dim):
                C[i][j] += A[i][k] * B[k][j]
    return C

def distribute_rows(A, size):
    """Distributes rows of matrix A (list of lists) to processes, handling uneven division."""
    total_rows = len(A)
    rows_per_process = [total_rows // size] * size
    for i in range(total_rows % size):
        rows_per_process[i] += 1

    displacements = [sum(rows_per_process[:i]) for i in range(size)]

    send_data = []
    for i in range(size):
        start_row = displacements[i]
        end_row = start_row + rows_per_process[i]
        send_data.append(A[start_row:end_row]) # A[start:end] works directly on lists
    return send_data, rows_per_process, displacements

def local_matrix_multiply(local_A_rows, B):
    """
    Performs matrix multiplication for a chunk of rows from A with full B.
    """
    num_local_rows = len(local_A_rows)
    if num_local_rows == 0:
        # Handle cases where a process gets 0 rows (e.g., N < size)
        # It should still return a valid empty matrix with correct column count
        # if B is known, or an empty list if N=0
        if B and len(B) > 0:
            return [[0 for _ in range(len(B[0]))] for _ in range(0)]
        return []

    common_dim = len(local_A_rows[0])
    cols_B = len(B[0])

    if common_dim != len(B):
        raise ValueError("Local A columns must match Matrix B rows for multiplication.")

    local_C_rows = [[0 for _ in range(cols_B)] for _ in range(num_local_rows)]

    for i in range(num_local_rows):
        for j in range(cols_B):
            for k in range(common_dim):
                local_C_rows[i][j] += local_A_rows[i][k] * B[k][j]
    return local_C_rows
# --- End Helper Functions ---


def run_serial_multiplication(N):
    """Runs the serial matrix multiplication and prints timing."""
    print(f"Running serial matrix multiplication for N={N} (pure Python)...")
    A = create_matrix(N, N)
    B = create_matrix(N, N)

    start_time = time.time()
    C = serial_matrix_multiply(A, B)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"TIME:{elapsed_time:.6f}") # Standardized output for easy parsing

    # --- Verification Section ---
    print("\n--- Verification (Serial) ---")
    expected_rows, expected_cols = N, N
    if len(C) == expected_rows and (len(C[0]) if C else 0) == expected_cols:
        print(f"Output matrix C has correct dimensions: {len(C)}x{len(C[0])}")
    else:
        print(f"ERROR: Output matrix C has incorrect dimensions: {len(C)}x{(len(C[0]) if C else 0)}, expected {expected_rows}x{expected_cols}")

    np_A, np_B = np.array(A), np.array(B)
    np_C_expected = np.dot(np_A, np_B)
    np_C_actual = np.array(C)

    if np.allclose(np_C_actual, np_C_expected):
        print("Pure Python result matches NumPy's np.dot result: SUCCESS")
    else:
        print("ERROR: Pure Python result does NOT match NumPy's np.dot result: FAILED")
    print("---------------------------\n")

def run_mpi_multiplication(N):
    """Runs the MPI distributed matrix multiplication and prints timing."""
    if not MPI_AVAILABLE:
        # This print will only happen if MPI_AVAILABLE is False,
        # which means mpi4py failed to import during script start.
        # It's less common when run with mpiexec, which sets up the environment.
        print("Error: mpi4py not found. Cannot run MPI version.")
        sys.exit(1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    A_full = None
    B_full = None
    C_full = None
    send_data_A = None

    if rank == 0:
        print(f"Running distributed matrix multiplication for N={N} with {size} processes (pure Python)...")
        A_full = create_matrix(N, N)
        B_full = create_matrix(N, N)
        
        send_data_A, _, _ = distribute_rows(A_full, size)

        start_time = MPI.Wtime()

    # Broadcast N and B_full
    N = comm.bcast(N, root=0)
    B_full = comm.bcast(B_full, root=0)

    # Scatter rows of A
    local_A_rows = comm.scatter(send_data_A, root=0)

    # Perform local multiplication
    local_C_rows = local_matrix_multiply(local_A_rows, B_full)

    # Gather results
    gathered_C_chunks = comm.gather(local_C_rows, root=0)

    if rank == 0:
        end_time = MPI.Wtime()
        elapsed_time = end_time - start_time
        print(f"TIME:{elapsed_time:.6f}") # Standardized output for easy parsing

        # Assemble C_full
        C_full = []
        for chunk in gathered_C_chunks:
            C_full.extend(chunk)

        # --- Verification Section ---
        print("\n--- Verification (Distributed Pure Python) ---")
        np_A_full = np.array(A_full)
        np_B_full = np.array(B_full)
        C_serial_check_np = np.dot(np_A_full, np_B_full)
        np_C_full_actual = np.array(C_full)

        if np.allclose(np_C_full_actual, C_serial_check_np):
            print("Distributed pure Python result matches NumPy serial verification: SUCCESS")
        else:
            print("ERROR: Distributed pure Python result does NOT match NumPy serial verification: FAILED")
            # Optional: Debugging information
            # diff = np.abs(np_C_full_actual - C_serial_check_np)
            # print("Max difference:", np.max(diff))

        expected_rows, expected_cols = N, N
        if len(C_full) == expected_rows and (len(C_full[0]) if C_full else 0) == expected_cols:
            print(f"Output matrix C_full has correct dimensions: {len(C_full)}x{len(C_full[0])}")
        else:
            print(f"ERROR: Output matrix C_full has incorrect dimensions: {len(C_full)}x{(len(C_full[0]) if C_full else 0)}, expected {expected_rows}x{expected_cols}")
        print("---------------------------\n")

    # MPI.Finalize() # This is handled automatically by mpi4py when the script exits

def main():
    # Parsing command-line arguments: --mode <serial|mpi> -N <matrix_dimension>
    if len(sys.argv) < 3:
        print("Usage: python matrix_multiplier.py --mode <serial|mpi> -N <matrix_dimension>")
        sys.exit(1)

    # Find the mode argument
    mode_arg_index = -1
    if "--mode" in sys.argv:
        mode_arg_index = sys.argv.index("--mode")
    
    if mode_arg_index == -1 or mode_arg_index + 1 >= len(sys.argv):
        print("Error: '--mode <serial|mpi>' argument is missing or incomplete.")
        sys.exit(1)
    
    mode = sys.argv[mode_arg_index + 1]

    # Find the -N argument
    n_arg_idx = -1
    if "-N" in sys.argv:
        n_arg_idx = sys.argv.index("-N")
    
    if n_arg_idx == -1 or n_arg_idx + 1 >= len(sys.argv):
        print("Error: '-N <matrix_dimension>' argument is missing or incomplete.")
        sys.exit(1)

    try:
        N = int(sys.argv[n_arg_idx + 1])
        if N <= 0:
            raise ValueError
    except ValueError:
        print("Error: N must be a positive integer.")
        sys.exit(1)

    if mode == "serial":
        # For serial execution, ensure we are not running under mpiexec by accident.
        # If MPI is initialized (e.g., via `mpiexec -n 1`), Get_size() will be 1.
        # The benchmark_runner will call this directly as `python matrix_multiplier.py ...`
        if MPI_AVAILABLE and MPI.Is_initialized() and MPI.COMM_WORLD.Get_size() > 1:
            print("Warning: Running serial mode with multiple MPI processes. This is not a true serial run.")
        run_serial_multiplication(N)
    elif mode == "mpi":
        if not MPI_AVAILABLE:
            print("Error: mpi4py not installed. Cannot run MPI mode.")
            sys.exit(1)
        # In MPI mode, this script MUST be run via 'mpiexec'.
        # The benchmark_runner takes care of this.
        run_mpi_multiplication(N)
    else:
        print("Error: Invalid mode. Use 'serial' or 'mpi'.")
        sys.exit(1)

if __name__ == "__main__":
    main()