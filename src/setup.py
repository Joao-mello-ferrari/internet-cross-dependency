import argparse
import os
from src.lib import run_script

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline setup')
    parser.add_argument("--country", type=str, required=True, help="Country code (label)")
    parser.add_argument("--code", type=str.lower, required=True, help="Country code (folder)")
    parser.add_argument('--semester', type=str, default="202501", help='Country code to read and write JSON files')
    parser.add_argument('--count', type=int, default=1000, help='Country code to read and write JSON files')
    args = parser.parse_args()


    domain_folder = os.path.join("results", args.code)
    os.makedirs(domain_folder, exist_ok=True)

    latency_output = os.path.join(domain_folder, "latency")
    os.makedirs(latency_output, exist_ok=True)
    
    locality_output = os.path.join(domain_folder, "locality")
    os.makedirs(locality_output, exist_ok=True)
    
    results_output = os.path.join(domain_folder, "results")
    os.makedirs(results_output, exist_ok=True)
    
    run_script(["make", "fetch_websites", f"COUNTRY=\"{args.country}\"", f"CODE=\"{args.code}\"", "QUERY=\"country\"", f"SEMESTER=\"{args.semester}\"", f"AMOUNT=\"{args.count}\""])
    run_script(["make", "classify_websites", f"COUNTRY=\"{args.country}\"", f"CODE=\"{args.code}\""])