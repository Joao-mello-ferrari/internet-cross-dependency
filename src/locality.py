import argparse
from src.lib import run_script

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline setup')
    parser.add_argument("--country", type=str.lower, required=True, help="Country code (label)")
    parser.add_argument("--code", type=str.lower, required=True, help="Country code (folder)")
    parser.add_argument('--vpn', type=str.lower, required=True, help='VPN node name used for data fetching')
    args = parser.parse_args()

    run_script(["make", "findcdn", f"COUNTRY=\"{args.country}\"", f"CODE=\"{args.code}\"", f"VPN=\"{args.vpn}\""])
    run_script(["make", "geolocate", f"COUNTRY=\"{args.country}\"", f"CODE=\"{args.code}\"", f"VPN=\"{args.vpn}\""])
    run_script(["make", "locedge", f"COUNTRY=\"{args.country}\"", f"CODE=\"{args.code}\"", f"VPN=\"{args.vpn}\""])