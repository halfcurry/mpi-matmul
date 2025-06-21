# mpi_matrix_multiply.py
from mpi4py import MPI
import random
import sys # Import the sys module
import numpy as np # Only for verification

def create_matrix(rows, cols):
    """Creates a matrix (list of lists) filled with random floats."""
    matrix = []
    for _ in range(rows):
        row = [random.random() for _ in range(cols)]
        matrix.append(row)
    return matrix

def distribute_rows(A, size):
    """
    Distributes rows of matrix A (list of lists) to processes, handling uneven division.
    This function calculates which rows each MPI process will receive.

    Args:
        A (list of lists): The full input matrix.
        size (int): The total number of MPI processes.

    Returns:
        tuple: A tuple containing:
            - send_data (list of list of lists): A list where each element is a
                                                 sub-matrix (chunk of rows) for one process.
            - rows_per_process (list of int): Number of rows assigned to each process.
            - displacements (list of int): Starting row index for each process's chunk
                                          in the original matrix.
    """
    total_rows = len(A)
    
    # 1. Calculate base number of rows for each process.
    #    Each process initially gets 'total_rows // size' rows.
    rows_per_process = [total_rows // size] * size

    # 2. Distribute remaining rows (if any) to the first few processes.
    #    If total_rows is not perfectly divisible by size, there will be a remainder.
    #    These extra rows are distributed one by one to the initial processes.
    for i in range(total_rows % size):
        rows_per_process[i] += 1

    # 3. Calculate displacement (starting row index) for each process.
    #    This helps to easily slice the original matrix.
    #    e.g., if rows_per_process is [3, 2, 2], displacements will be [0, 3, 5]
    displacements = [sum(rows_per_process[:i]) for i in range(size)]

    # 4. Create the chunks of data (sub-matrices) to be sent.
    #    Each element in send_data is a slice (list of lists) from the original matrix A.
    send_data = []
    for i in range(size):
        start_row = displacements[i]
        end_row = start_row + rows_per_process[i]
        send_data.append(A[start_row:end_row]) # Slices matrix A to get the chunk for process 'i'

    return send_data, rows_per_process, displacements

def local_matrix_multiply(local_A_rows, B):
    """
    Performs matrix multiplication for a chunk of rows from A with full B.
    local_A_rows is num_local_rows x common_dim
    B is common_dim x cols_B
    Returns a list of lists representing the local result.
    """
    num_local_rows = len(local_A_rows)
    if num_local_rows == 0:
        return [] # Handle empty chunks

    common_dim = len(local_A_rows[0])
    cols_B = len(B[0])

    if common_dim != len(B):
        raise ValueError("Local A columns must match Matrix B rows for multiplication.")

    local_C_rows = [[0 for _ in range(cols_B)] for _ in range(num_local_rows)]

    for i in range(num_local_rows): # Iterate over rows of local_A_rows
        for j in range(cols_B):     # Iterate over columns of B
            for k in range(common_dim): # Iterate over common dimension
                local_C_rows[i][j] += local_A_rows[i][k] * B[k][j]
    return local_C_rows


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- Read N from command-line arguments on root process ---
    if rank == 0:
        if len(sys.argv) < 2:
            print("Usage: mpiexec -n <num_processes> python mpi_matrix_multiply.py <matrix_dimension_N>")
            comm.Abort(1) # Abort all processes
        try:
            N = int(sys.argv[1])
            if N <= 0:
                raise ValueError
        except ValueError:
            print("Error: N must be a positive integer.")
            comm.Abort(1) # Abort all processes
    else:
        N = None # Initialize N for other processes

    # Broadcast N to all processes
    N = comm.bcast(N, root=0)
    # --------------------------------------------------------

    A_full = None
    B_full = None
    C_full = None # This will hold the gathered result on root (list of lists)
    C_serial_check_np = None # For NumPy verification on root
    
    send_data_A = None # For scatter

    if rank == 0:
        print(f"Running distributed matrix multiplication for N={N} with {size} processes (pure Python)...")
        # Root process initializes matrices (pure Python lists of lists)
        A_full = create_matrix(N, N)
        B_full = create_matrix(N, N)
        
        # Prepare send_data for scatter
        send_data_A, rows_per_process, displacements = distribute_rows(A_full, size)

        start_time = MPI.Wtime() # Start timing for the root process

    # 1. Broadcast the full matrix B to all processes
    B_full = comm.bcast(B_full, root=0) # B is a list of lists

    # 2. Scatter rows of A
    local_A_rows = comm.scatter(send_data_A, root=0)

    # 3. Perform local matrix multiplication (pure Python)
    local_C_rows = local_matrix_multiply(local_A_rows, B_full)

    # 4. Gather results back to the root process
    gathered_C_chunks = comm.gather(local_C_rows, root=0)

    if rank == 0:
        end_time = MPI.Wtime() # End timing for the root process
        
        # Assemble the final matrix C from gathered chunks
        C_full = []
        for chunk in gathered_C_chunks:
            C_full.extend(chunk) # Extend the list of lists with rows from each chunk

        print(f"Distributed execution time ({size} processes): {end_time - start_time:.4f} seconds")

        # --- Verification Section (on root process) ---
        print("\n--- Verification (Distributed Pure Python) ---")
        # 1. Compute serial result using NumPy for ground truth
        print("Performing NumPy-based serial verification on root process...")
        np_A_full = np.array(A_full) # Convert original Python list to NumPy
        np_B_full = np.array(B_full) # Convert original Python list to NumPy
        C_serial_check_np = np.dot(np_A_full, np_B_full) # NumPy's ground truth

        # 2. Convert our distributed result to NumPy for comparison
        np_C_full_actual = np.array(C_full)

        # 3. Compare the distributed result with the serial NumPy result
        if np.allclose(np_C_full_actual, C_serial_check_np):
            print("Distributed pure Python result matches NumPy serial verification: SUCCESS")
        else:
            print("ERROR: Distributed pure Python result does NOT match NumPy serial verification: FAILED")
            # Optional: Debugging information
            # diff = np.abs(np_C_full_actual - C_serial_check_np)
            # print("Max difference:", np.max(diff))
            # print("Elements not close:", np.where(~np.isclose(np_C_full_actual, C_serial_check_np)))

        # 4. Check dimensions of the result
        expected_rows = N
        expected_cols = N
        if len(C_full) == expected_rows and (len(C_full[0]) if C_full else 0) == expected_cols:
            print(f"Output matrix C_full has correct dimensions: {len(C_full)}x{len(C_full[0])}")
        else:
            print(f"ERROR: Output matrix C_full has incorrect dimensions: {len(C_full)}x{(len(C_full[0]) if C_full else 0)}, expected {expected_rows}x{expected_cols}")

        # 5. For very small matrices, print a sample
        if N <= 5:
            print("\nSample of Distributed C (full matrix):\n", C_full)
            print("\nSample of Serial Verification C (full matrix via NumPy):\n", C_serial_check_np)
        else:
            print("\nSample of Distributed C (top-left 5x5):\n", [row[:5] for row in C_full[:5]])
            print("\nSample of Serial Verification C (top-left 5x5 via NumPy):\n", C_serial_check_np[:5, :5])
        print("---------------------------\n")

    MPI.Finalize()