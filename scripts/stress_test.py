import subprocess
import sys
import time
from typing import List, Tuple

def run_command(cmd: List[str], desc: str) -> Tuple[bool, str, float]:
    print(f"Testing: {desc}...")
    start = time.time()
    try:
        # Run process
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=False,
            encoding='utf-8'
        )
        duration = time.time() - start
        
        if result.returncode == 0:
            print(f"  ✅ PASS ({duration:.2f}s)")
            return True, result.stdout, duration
        else:
            print(f"  ❌ FAIL (Return Code: {result.returncode})")
            print(f"  Error: {result.stderr[:200]}...")
            return False, result.stderr, duration
            
    except Exception as e:
        print(f"  ❌ CRITICAL FAIL: {e}")
        return False, str(e), 0.0

def main():
    print("=== TENNIS APP STRESS TEST ===")
    
    python = sys.executable
    script = "tennis.py"
    
    tests = [
        # 1. Basic CLI
        ([python, script, "--help"], "Help Command"),
        ([python, script, "scrape", "--help"], "Scrape Help"),
        
        # 2. Argument Validation (Should fail gracefully, but return error code)
        ([python, script, "scrape", "players"], "Missing IDs (Expect Fail)"),
        ([python, script, "scrape", "upcoming", "--days", "-1"], "Negative Days (Edge Case)"),
        
        # 3. Functional Tests (Light)
        ([python, script, "scrape", "upcoming", "--days", "1"], "Scrape Upcoming (1 Day)"),
        ([python, script, "predict", "--days", "1", "--no-scrape"], "Predict (No Scrape)"),
        
        # 4. Stress / Load (Simulated)
        ([python, script, "predict", "--min-odds", "10.0"], "Predict High Odds Filter"),
        ([python, script, "predict", "--confidence", "0.99"], "Predict High Confidence"),
    ]
    
    passed = 0
    total = len(tests)
    results = []
    
    for cmd, desc in tests:
        success, output, dur = run_command(cmd, desc)
        results.append({
            "test": desc,
            "success": success,
            "duration": dur
        })
        if success:
            passed += 1
            
    print("\n=== SUMMARY ===")
    print(f"Passed: {passed}/{total}")
    
    if passed < total:
        print("\nFailures:")
        for r in results:
            if not r["success"]:
                print(f"- {r['test']}")

if __name__ == "__main__":
    main()
