# benchmark_runner.py
import subprocess
import re
import sys
import numpy as np # For average, std dev, speedup calculations

def run_test(mode, N, num_processes=None):
    """
    Runs the matrix_multiplier.py script and captures its execution time.
    Returns the time in seconds.
    """
    command = []
    if mode == "mpi":
        if num_processes is None:
            raise ValueError("num_processes must be specified for MPI mode.")
        # --- THIS IS THE LINE TO FIX THE MPI ERROR ---
        command = ["mpiexec", "--oversubscribe", "-n", str(num_processes), "python", "matrix_multiplier.py", "--mode", "mpi", "-N", str(N)]
        # --------------------------------------------
    elif mode == "serial":
        command = ["python", "matrix_multiplier.py", "--mode", "serial", "-N", str(N)]
    else:
        raise ValueError("Invalid mode. Use 'serial' or 'mpi'.")

    print(f"Executing: {' '.join(command)}")
    # Use check=True to raise an error if the subprocess fails
    process = subprocess.run(command, capture_output=True, text=True, check=True)

    # Extract time from stdout using regex
    match = re.search(r"TIME:(\d+\.\d+)", process.stdout)
    if match:
        return float(match.group(1))
    else:
        print("ERROR: Could not find 'TIME:' in output.")
        print("Stdout:\n", process.stdout)
        print("Stderr:\n", process.stderr)
        raise RuntimeError("Failed to parse time from script output.")

def main():
    if len(sys.argv) < 3:
        print("Usage: python benchmark_runner.py <matrix_dimension_N> <num_runs>")
        sys.exit(1)

    try:
        N = int(sys.argv[1])
        num_runs = int(sys.argv[2])
        if N <= 0 or num_runs <= 0:
            raise ValueError
    except ValueError:
        print("Error: Matrix dimension N and number of runs must be positive integers.")
        sys.exit(1)

    print(f"\n--- Benchmarking Matrix Multiplication (N={N}, Runs per test={num_runs}) ---")
    print("WARNING: Pure Python matrix multiplication is very slow. Use small N values (e.g., 50-150).\n")

    results = {}

    # --- Serial Benchmarking ---
    print(f"\n--- Running Serial Benchmarks (N={N}) ---")
    serial_times = []
    for i in range(num_runs):
        try:
            time_taken = run_test("serial", N)
            serial_times.append(time_taken)
            print(f"Serial Run {i+1}: {time_taken:.6f} seconds")
        except Exception as e:
            print(f"Error during serial run {i+1}: {e}")
            break # Stop if an error occurs

    if serial_times:
        avg_serial_time = np.mean(serial_times)
        std_dev_serial_time = np.std(serial_times)
        results["Serial"] = {"avg_time": avg_serial_time, "std_dev": std_dev_serial_time}
        print(f"\nAverage Serial Time ({num_runs} runs): {avg_serial_time:.6f} +/- {std_dev_serial_time:.6f} seconds")
    else:
        print("No successful serial runs.")
        results["Serial"] = {"avg_time": float('inf'), "std_dev": float('inf')} # Indicate failure/no data

    # --- MPI Benchmarking ---
    # Adjust mpi_process_counts based on your Codespace cores (e.g., 1, 2, 3, 4 for 4 cores)
    mpi_process_counts = [1, 2, 3, 4] 
    for p in mpi_process_counts:
        print(f"\n--- Running MPI Benchmarks (N={N}, Processes={p}) ---")
        mpi_times = []
        for i in range(num_runs):
            try:
                time_taken = run_test("mpi", N, num_processes=p)
                mpi_times.append(time_taken)
                print(f"MPI {p} Processes Run {i+1}: {time_taken:.6f} seconds")
            except Exception as e:
                print(f"Error during MPI run {i+1} with {p} processes: {e}")
                break # Stop if an error occurs
        
        if mpi_times:
            avg_mpi_time = np.mean(mpi_times)
            std_dev_mpi_time = np.std(mpi_times)
            results[f"MPI {p} Processes"] = {"avg_time": avg_mpi_time, "std_dev": std_dev_mpi_time}
            print(f"\nAverage MPI Time ({p} Processes, {num_runs} runs): {avg_mpi_time:.6f} +/- {std_dev_mpi_time:.6f} seconds")
        else:
            print(f"No successful MPI runs for {p} processes.")
            results[f"MPI {p} Processes"] = {"avg_time": float('inf'), "std_dev": float('inf')}

    # --- Summary and Comparison ---
    print("\n--- Benchmark Summary ---")
    print(f"{'Mode':<20} | {'Avg Time (s)':<15} | {'Std Dev (s)':<15} | {'Speedup':<10} | {'Efficiency':<10}")
    print("-" * 80)

    serial_avg = results["Serial"]["avg_time"]

    for mode_name, data in results.items():
        avg_time = data["avg_time"]
        std_dev = data["std_dev"]
        
        speedup = "N/A"
        efficiency = "N/A"
        if serial_avg != float('inf') and avg_time != float('inf') and avg_time != 0:
            speedup_val = serial_avg / avg_time
            speedup = f"{speedup_val:.2f}x"
            if "MPI" in mode_name:
                p_match = re.search(r"MPI (\d+) Processes", mode_name)
                if p_match:
                    num_p = int(p_match.group(1))
                    if num_p > 0:
                        efficiency = f"{(speedup_val / num_p * 100):.2f}%"


        print(f"{mode_name:<20} | {avg_time:<15.6f} | {std_dev:<15.6f} | {speedup:<10} | {efficiency:<10}")

    print("\nBenchmarking complete.")

if __name__ == "__main__":
    main()