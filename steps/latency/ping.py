import os
import re
import json
import time
import argparse
import requests
from dotenv import load_dotenv
from ping3 import ping
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from steps.latency.fetch_probes import fetch_probes
from steps.latency.helpers import find_ipv6_has_address

# ====================
# Constants
# ====================
RIPE_ATLAS_MEASUREMENTS_URL = "https://atlas.ripe.net/api/v2/measurements/"
RIPE_ATLAS_RESULTS_URL = "https://atlas.ripe.net/api/v2/measurements/{}/results?format=json"


# ====================
# Helper Functions
# ====================
def save_json(data, filepath):
    os.makedirs(filepath.parent, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def get_probes_list(country_ases, af):
    key = "v4" if af == 4 else "v6"
    probes = []
    for country_as in country_ases.values():
        probes.extend(country_as.get(key, []))
    return probes


def create_ping_measurement(domain, country_ases, af):
    domain = domain.replace("http://", "").replace("https://", "")
    accepts_icmp = ping(domain, timeout=1)

    # Serves block icmp requests
    if not accepts_icmp:
        return {}, domain, None

    # Not ipv6 could be established
    if af == 6 and not find_ipv6_has_address(domain):
        return {}, None, domain

    probes = get_probes_list(country_ases, af)
    if not probes:
        raise Exception(f"No probes found for AF{af} in {domain}")

    headers = {
        "Authorization": f"Key {os.getenv('RIPE_ATLAS_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "definitions": [{
            "type": "ping",
            "af": af,
            "resolve_on_probe": True,
            "description": f"Ping to {domain} AF{af}",
            "packets": 3,
            "size": 48,
            "skip_dns_check": False,
            "include_probe_id": False,
            "target": domain
        }],
        "probes": [{
            "type": "probes",
            "value": ",".join(map(str, probes)),
            "requested": len(probes)
        }],
        "is_oneoff": True,
        "bill_to": os.getenv("RIPE_ATLAS_BILL_TO_EMAIL"),
        "start_time": int(time.time()) + 10
    }

    resp = requests.post(RIPE_ATLAS_MEASUREMENTS_URL, headers=headers, json=payload)

    if resp.ok:
        return {domain: resp.json().get("measurements")}, None, None
    else:
        raise Exception(f"Failed to create AF{af} measurement for {domain}: {resp.status_code} {resp.text}")


def fetch_measurement_result(measurement_id):
    resp = requests.get(RIPE_ATLAS_RESULTS_URL.format(measurement_id))
    resp.raise_for_status()
    return resp.json()


def parse_rtt_results(response):
    sorted_by_probe = sorted(response, key=lambda x: x.get("prb_id"))
    results = []
    for entry in sorted_by_probe:
        probe_results = entry.get("result", [])
        rtts = [m.get("rtt") for m in probe_results if m.get("rtt") is not None]
        results.append(rtts)
    return results


# Fetch RTT results
def fetch_all_rtts(measurements, proto, log):
    rtts = {}
    for site, ids in tqdm(measurements.items(), desc=f"Fetching {proto.upper()} RTTs"):
        try:
            data = fetch_measurement_result(ids[0])
            parsed = parse_rtt_results(data)
            rtts[site] = parsed
            log_entry = f"{proto.upper()} {site} ID {ids[0]}: {data}\n\n"
            log += log_entry
        except Exception as e:
            log += f"Error fetching {proto} {site}: {e}\n"
            print(e)
    return rtts, log



# ====================
# Main Execution
# ====================
def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ping measurement with RIPE Atlas.")
    parser.add_argument("--country", type=str.lower, required=True, help="Country name")
    parser.add_argument("--code", type=str.lower, required=True, help="Country code")
    parser.add_argument("--fetch_fresh_probes", action="store_true", help="Fetch fresh probes if set")
    args = parser.parse_args()

    # Setup paths
    base_path = Path(f"results/{args.code}")
    input_file = base_path / "output.json"
    probes_file = base_path / "probes.json"
    latency_path = base_path / "latency"
    latency_path.mkdir(parents=True, exist_ok=True)

    # Output files
    output_files = {
        "latency_v4": latency_path / "latency_ipv4.json",
        "latency_v6": latency_path / "latency_ipv6.json",
        "icmp_block": latency_path / "icmp_block.json",
        "fail_ipv6": latency_path / "fail_ipv6_route.json",
        "log": latency_path / "log.txt",
        "raw": latency_path / "raw_measurements.json"
    }

    # Load input data
    with open(input_file) as f:
        websites = json.load(f)

    # Load or fetch probes
    if args.fetch_fresh_probes:
        fetch_probes(args.code)

    with open(probes_file) as f:
        probes = json.load(f).get("ases", {})

    # Measurement containers
    results_v4, results_v6 = {}, {}
    icmp_block, ipv6_fail, log = [], [], ""

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(create_ping_measurement, site, probes, af): (proto, site)
            for site in websites
            for af, proto in [(4, "v4"), (6, "v6")]
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Running measurements"):
            proto, site = futures[future]
            try:
                measurement_id, icmp_block_domain, fail_ipv6_route_domain = future.result()
                if proto == "v4":
                    results_v4.update(measurement_id)
                else:
                    results_v6.update(measurement_id)

                if icmp_block_domain:
                    icmp_block.append(icmp_block_domain)
                if fail_ipv6_route_domain:
                    ipv6_fail.append(fail_ipv6_route_domain)
            except Exception as e:
                log += f"Error {proto} {site}: {e}\n"
                print(e)


    # Wait for measurements to be available
    for i in range(10):
        print(f"Waiting... {10 * (i + 1)} seconds elapsed")
        time.sleep(10)

    

    ipv4_rtts, log = fetch_all_rtts(results_v4, "v4", log)
    ipv6_rtts, log = fetch_all_rtts(results_v6, "v6", log)

    # Save results
    save_json(ipv4_rtts, output_files["latency_v4"])
    save_json(ipv6_rtts, output_files["latency_v6"])
    save_json(ipv6_fail, output_files["fail_ipv6"])
    save_json(icmp_block, output_files["icmp_block"])
    save_json({"v4": results_v4, "v6": results_v6}, output_files["raw"])

    with open(output_files["log"], "w") as f:
        f.write(log)

    print(f"\n✅ Measurement complete. Results saved to {latency_path}")


if __name__ == "__main__":
    main()
