import os
import json
import time
import argparse
import requests
from dotenv import load_dotenv
from ping3 import ping
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from math import ceil

from src.steps.latency.fetch_probes import fetch_probes
from src.steps.latency.create_measurements import create_ping_measurement


RIPE_ATLAS_RESULTS_URL = "https://atlas.ripe.net/api/v2/measurements/{}/results?format=json"
BATCH_SIZE = 50
SLEEP_TIME_TO_SCHEDULE_SECONDS = 180
SLEEP_TIME_TO_SCHEDULE_SECONDS_SINGLE = 2.5

def save_json(data, filepath):
    os.makedirs(filepath.parent, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def fetch_measurement_result(measurement_id, attempts=3):
    try:
        resp = requests.get(RIPE_ATLAS_RESULTS_URL.format(measurement_id), timeout=2)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        print("Request timed out! Retrying...")
        if attempts > 1:
            time.sleep(2*(4-attempts))
            return fetch_measurement_result(measurement_id, attempts - 1)
        raise Exception("Request timed out while fetching measurement results.")

def parse_rtt_results(response):
    sorted_by_probe = sorted(response, key=lambda x: x.get("prb_id"))
    results = []
    results_with_probe_id = {}
    for entry in sorted_by_probe:
        probe_results = entry.get("result", [])
        rtts = [m.get("rtt") for m in probe_results if m.get("rtt") is not None]
        results.append(rtts)
        results_with_probe_id[entry.get("prb_id")] = rtts
    return results, results_with_probe_id

def fetch_all_rtts(measurements, proto, log):
    rtts = {}
    rtts_with_probe_id = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(fetch_measurement_result, ids[0]): (site, ids[0])
            for site, ids in measurements.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Fetching {proto.upper()} RTTs"):
            site, measurement_id = futures[future]
            try:
                data = future.result()
                parsed, parsed_with_probe_id = parse_rtt_results(data)
                rtts[site] = parsed
                rtts_with_probe_id[site] = parsed_with_probe_id
                log_entry = f"{proto.upper()} {site} ID {measurement_id}: {data}\n\n"
                log += log_entry
            except Exception as e:
                log += f"Error fetching {proto} {site}: {e}\n"
                print(e)
    return rtts, rtts_with_probe_id, log

def fetch_all_rtts_single(measurements, proto, log):
    rtts = {}
    for site, ids in tqdm(measurements.items(), desc="Fetching measurements for af {}".format(proto)):
        measurement_id = ids[0]
        try:
            data = fetch_measurement_result(measurement_id)
            parsed, _ = parse_rtt_results(data)
            rtts[site] = parsed
            log_entry = f"{proto.upper()} {site} ID {measurement_id}: {data}\n\n"
            log += log_entry
        except Exception as e:
            log += f"Error fetching {proto} {site}: {e}\n"
            print(e)
    return rtts, log

def batch_process(websites, probes, af):
    results, icmp_block, ipv6_fail, log = {}, [], [], ""
    for i in range(0, len(websites), BATCH_SIZE):
        batch = websites[i:i + BATCH_SIZE]
        print(f"\nSubmitting batch {i//BATCH_SIZE + 1} of {ceil(len(websites)/BATCH_SIZE)} for AF{af}")
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(create_ping_measurement, site, probes, af): site
                for site in batch
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Running measurements"):
                site = futures[future]
                try:
                    measurement_id, icmp_block_domain, fail_ipv6_route_domain = future.result()
                    results.update(measurement_id)
                    if icmp_block_domain:
                        icmp_block.append(icmp_block_domain)
                    if fail_ipv6_route_domain:
                        ipv6_fail.append(fail_ipv6_route_domain)
                except Exception as e:
                    log += f"Error AF{af} {site}: {e}\n"
                    print(e)
        print(f"⏳ Waiting {SLEEP_TIME_TO_SCHEDULE_SECONDS} seconds for batch measurements to complete...")
        time.sleep(SLEEP_TIME_TO_SCHEDULE_SECONDS)
    return results, icmp_block, ipv6_fail, log

def single_process(websites, probes, af):
    results, icmp_block, ipv6_fail, log = {}, [], [], ""
    for site in tqdm(websites, desc="Running measurements for af {}".format(af)):
        try: 
            measurement_id, icmp_block_domain, fail_ipv6_route_domain = create_ping_measurement(site, probes, af)
            results.update(measurement_id)
            if icmp_block_domain:
                icmp_block.append(icmp_block_domain)
            if fail_ipv6_route_domain:
                ipv6_fail.append(fail_ipv6_route_domain)
        except Exception as e:
            log += f"Error AF{af} {site}: {e}\n"
            print(e)
        time.sleep(SLEEP_TIME_TO_SCHEDULE_SECONDS_SINGLE)
    return results, icmp_block, ipv6_fail, log

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Batch ping measurement with RIPE Atlas.")
    parser.add_argument("--country", type=str.lower, required=True)
    parser.add_argument("--code", type=str.lower, required=True)
    parser.add_argument("--fetch_fresh_probes", action="store_true")
    args = parser.parse_args()

    probes_file = Path("results/probes.json")
    base_path = Path(f"results/{args.code}")
    input_file = base_path / "output.json"
    latency_path = base_path / "latency"
    latency_path.mkdir(parents=True, exist_ok=True)

    output_files = {
        "latency_v4": latency_path / "latency_ipv4.json",
        "latency_v4_with_probe_id": latency_path / "latency_ipv4_with_probe_id.json",
        "latency_v6": latency_path / "latency_ipv6.json",
        "latency_v6_with_probe_id": latency_path / "latency_ipv6_with_probe_id.json",
        "icmp_block": latency_path / "icmp_block.json",
        "fail_ipv6": latency_path / "fail_ipv6_route.json",
        "log": latency_path / "log.txt",
        "raw": latency_path / "raw_measurements.json"
    }

    with open(input_file) as f:
        websites = json.load(f)

    if args.fetch_fresh_probes:
        try:
            fetch_probes(args.code)
            probes_file = base_path / "probes.json"
        except: 
            print("Failed to fetch fresh probes, using existing probes file.")
    with open(probes_file) as f:
        probes = json.load(f).get("ases", {})

    # --------------------------------
    # file = open(output_files["raw"], "r")
    # file = json.load(file)
    # results_v4 = file["v4"]
    # results_v6 = file["v6"]
    # log_v4 = ""
    # log_v6 = ""

    results_v4, icmp_block_v4, _, log_v4 = single_process(websites, probes, af=4)
    results_v6, _, ipv6_fail_v6, log_v6 = single_process(websites, probes, af=6)

    time.sleep(30)
    # --------------------------------

    ipv4_rtts, ipv4_rtts_with_probe_id, log_v4 = fetch_all_rtts(results_v4, "v4", log_v4)
    ipv6_rtts, ipv6_rtts_with_probe_id, log_v6 = fetch_all_rtts(results_v6, "v6", log_v6)

    save_json(ipv4_rtts, output_files["latency_v4"])
    save_json(ipv4_rtts_with_probe_id, output_files["latency_v4_with_probe_id"])
    save_json(ipv6_rtts, output_files["latency_v6"])
    save_json(ipv6_rtts_with_probe_id, output_files["latency_v6_with_probe_id"])
    save_json(ipv6_fail_v6, output_files["fail_ipv6"])
    save_json(icmp_block_v4, output_files["icmp_block"])
    save_json({"v4": results_v4, "v6": results_v6}, output_files["raw"])

    with open(output_files["log"], "w") as f:
        f.write(log_v4 + log_v6)

    print(f"\n✅ Measurement complete. Results saved to {latency_path}")

if __name__ == "__main__":
    main()
