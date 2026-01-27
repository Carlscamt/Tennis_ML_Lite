
import subprocess
import time
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, stdev

def run_command(cmd):
    """Run a shell command and return execution time and success status."""
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        duration = time.time() - start_time
        return True, duration, result.stdout
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        return False, duration, e.stderr

def stress_test(command, iterations, concurrency=1):
    print(f"Starting stress test for command: '{command}'")
    print(f"Iterations: {iterations}, Concurrency: {concurrency}")
    
    results = []
    success_count = 0
    failure_count = 0
    
    start_total = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_command, command) for _ in range(iterations)]
        
        for i, future in enumerate(as_completed(futures)):
            success, duration, output = future.result()
            results.append(duration)
            
            if success:
                success_count += 1
                status = "SUCCESS"
            else:
                failure_count += 1
                status = "FAILURE"
                print(f"run {i+1} failed: {output[:200]}...") # Print first 200 chars of error
            
            print(f"Run {i+1}/{iterations}: {status} in {duration:.2f}s")

    total_time = time.time() - start_total
    
    if not results:
        print("No results collected.")
        return

    avg_time = mean(results)
    min_time = min(results)
    max_time = max(results)
    std_dev = stdev(results) if len(results) > 1 else 0.0

    print("\n" + "="*40)
    print("STRESS TEST RESULTS")
    print("="*40)
    print(f"Total Wall Time: {total_time:.2f}s")
    print(f"Success Rate:    {success_count}/{iterations} ({success_count/iterations*100:.1f}%)")
    print(f"Avg Duration:    {avg_time:.2f}s +/- {std_dev:.2f}s")
    print(f"Min Duration:    {min_time:.2f}s")
    print(f"Max Duration:    {max_time:.2f}s")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test CLI command")
    parser.add_argument("--cmd", default="python tennis.py predict --days 1", help="Command to run")
    parser.add_argument("--iter", type=int, default=5, help="Number of iterations")
    parser.add_argument("--parallel", type=int, default=1, help="Concurrency level")
    
    args = parser.parse_args()
    
    stress_test(args.cmd, args.iter, args.parallel)
