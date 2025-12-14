import argparse
from src.lib import run_script

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline setup')
    parser.add_argument("--country", type=str, required=True, help="Country code (label)")
    parser.add_argument("--code", type=str.lower, required=True, help="Country code (folder)")
    args = parser.parse_args()

    # Latency analysis for given country
    run_script(["make", "analysis/latency/cdf", f"COUNTRY=\"{args.country}\"", f"CODE=\"{args.code}\"", f"SAVE=\"TRUE\""])
    run_script(["make", "analysis/latency/difference", f"COUNTRY=\"{args.country}\"", f"CODE=\"{args.code}\"", f"SAVE=\"TRUE\""])
    run_script(["make", "analysis/latency/prob_distribution", f"COUNTRY=\"{args.country}\"", f"CODE=\"{args.code}\"", f"SAVE=\"TRUE\""])
    
    # Locality analysis for given country
    run_script(["make", "analysis/locality/country", f"COUNTRY=\"{args.country}\"", f"CODE=\"{args.code}\"", f"SAVE=\"TRUE\"", f"ACCUMULATED=\"TRUE\""])
    run_script(["make", "analysis/locality/country", f"COUNTRY=\"{args.country}\"", f"CODE=\"{args.code}\"", f"SAVE=\"TRUE\""])
    run_script(["make", "analysis/locality/cdn_provider", f"COUNTRY=\"{args.country}\"", f"CODE=\"{args.code}\"", f"SAVE=\"TRUE\"", f"ACCUMULATED=\"TRUE\""])
    run_script(["make", "analysis/locality/cdn_provider", f"COUNTRY=\"{args.country}\"", f"CODE=\"{args.code}\"", f"SAVE=\"TRUE\""])
