# serial_matrix_multiply.py
import time
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

if __name__ == "__main__":
    # --- Read N from command-line arguments ---
    if len(sys.argv) < 2:
        print("Usage: python serial_matrix_multiply.py <matrix_dimension_N>")
        sys.exit(1) # Exit with an error code
    try:
        N = int(sys.argv[1])
        if N <= 0:
            raise ValueError
    except ValueError:
        print("Error: N must be a positive integer.")
        sys.exit(1)
    # ----------------------------------------

    print(f"Running serial matrix multiplication for N={N} (pure Python)...")

    # Generate random matrices using pure Python
    A = create_matrix(N, N)
    B = create_matrix(N, N)

    start_time = time.time()
    C = serial_matrix_multiply(A, B)
    end_time = time.time()

    print(f"Serial execution time: {end_time - start_time:.4f} seconds")

    # --- Verification Section ---
    print("\n--- Verification (Serial) ---")
    # 1. Check dimensions of the result
    expected_rows = N
    expected_cols = N
    if len(C) == expected_rows and (len(C[0]) if C else 0) == expected_cols:
        print(f"Output matrix C has correct dimensions: {len(C)}x{len(C[0])}")
    else:
        print(f"ERROR: Output matrix C has incorrect dimensions: {len(C)}x{(len(C[0]) if C else 0)}, expected {expected_rows}x{expected_cols}")

    # 2. Verify correctness using NumPy for comparison
    # Convert lists of lists to NumPy arrays for efficient comparison
    np_A = np.array(A)
    np_B = np.array(B)
    np_C_expected = np.dot(np_A, np_B) # NumPy's ground truth

    np_C_actual = np.array(C) # Our Python result as NumPy array

    if np.allclose(np_C_actual, np_C_expected):
        print("Pure Python result matches NumPy's np.dot result: SUCCESS")
    else:
        print("ERROR: Pure Python result does NOT match NumPy's np.dot result: FAILED")
        # Optional: Debugging information
        # diff = np.abs(np_C_actual - np_C_expected)
        # print("Max difference:", np.max(diff))

    # 3. For very small matrices, print a sample
    if N <= 5:
        print("\nSample of A:\n", A)
        print("\nSample of B:\n", B)
        print("\nSample of C (full matrix):\n", C)
    else:
        print("\nSample of C (top-left 5x5):\n", [row[:5] for row in C[:5]])
    print("---------------------------\n")