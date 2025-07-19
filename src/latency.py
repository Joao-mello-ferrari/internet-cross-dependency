import argparse
from src.lib import run_script

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline setup')
    parser.add_argument("--country", type=str, required=True, help="Country code (label)")
    parser.add_argument("--code", type=str.lower, required=True, help="Country code (folder)")
    args = parser.parse_args()

    run_script(["make", "ping", f"COUNTRY=\"{args.country}\"", f"CODE=\"{args.code}\"", f"FETCH_FRESH_PROBES=\"--fetch_fresh_probes\""])
    #run_script(["make", "ping", f"COUNTRY=\"{args.country}\"", f"CODE=\"{args.code}\""])