import requests
import re
from helpers import run_command
from ping3 import ping

def find_ipv6_has_address(hostname):
    results = run_command(["dig", "+short", hostname, "AAAA"])
    # Use regex to extract the first IPv6 address from the results
    ipv6_pattern = r'\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}|\b(?:[0-9a-fA-F]{1,4}:){1,7}:|\b:(:[0-9a-fA-F]{1,4}){1,7}\b'
    for result in results.splitlines():
        _match = re.search(ipv6_pattern, result)
        if _match:
            return True
    return False

def create_ping_measurement_v4(domain, country_ases):
    _domain = domain.replace("http://", "").replace("https://", "")
    accepts_icmp = ping(_domain, timeout=1)
    
    if not accepts_icmp:
        return {}, [_domain]

    api_key = "1cf1f929-9e31-4bae-ac1f-6487a82d8daa"
    bill_to_email = "joao.vico.mellof@gmail.com"
    start_time = int(time.time()) + 10
    url = "https://atlas.ripe.net/api/v2/measurements/"
    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json"
    }
    
    probesv4 = []
    for country_as in country_ases.values():
        probesv4.extend(country_as.get("v4", []))
    
    if not probesv4:
        raise Exception(f"No probes found for country ASes in {domain}")

    datav4 = {
        "definitions": [
            {
                "type": "ping",
                "af": 4,
                "resolve_on_probe": True,
                "description": f"Ping measurement to {_domain}",
                "packets": 3,
                "size": 48,
                "skip_dns_check": False,
                "include_probe_id": False,
                "target": _domain
            }
        ],
        "probes": [
            {
                "type": "probes",
                "value": ",".join(map(str, probesv4)),
                "requested": len(probesv4)
            }
        ],
        "is_oneoff": True,
        "bill_to": bill_to_email,
        "start_time": start_time
    }

    response = requests.post(url, headers=headers, json=datav4)
    if 200 <= response.status_code < 300:
        return { _domain: sorted(response.json().get("measurements")) }, []
    else:
        raise Exception(f"Failed to create IPv4 measurement for {_domain}: {response.status_code} {response.text}")


def create_ping_measurement_v6(domain, country_ases):
    _domain = domain.replace("http://", "").replace("https://", "")
    accepts_icmp = ping(_domain, timeout=1)

    if not find_ipv6_has_address(_domain) or not accepts_icmp:
        return {}, [_domain]  # No IPv6 address, log failure

    api_key = "1cf1f929-9e31-4bae-ac1f-6487a82d8daa"
    bill_to_email = "joao.vico.mellof@gmail.com"
    start_time = int(time.time()) + 10
    url = "https://atlas.ripe.net/api/v2/measurements/"
    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json"
    }
    
    probesv6 = []
    for country_as in country_ases.values():
        probesv6.extend(country_as.get("v6", []))
    
    if not probesv6:
        raise Exception(f"No probes found for country ASes in {domain}")

    datav6 = {
        "definitions": [
            {
                "type": "ping",
                "af": 6,
                "resolve_on_probe": True,
                "description": f"Ping measurement to {_domain}",
                "packets": 3,
                "size": 48,
                "skip_dns_check": False,
                "include_probe_id": False,
                "target": _domain
            }
        ],
        "probes": [
            {
                "type": "probes",
                "value": ",".join(map(str, probesv6)),
                "requested": len(probesv6)
            }
        ],
        "is_oneoff": True,
        "bill_to": bill_to_email,
        "start_time": start_time
    }

    response = requests.post(url, headers=headers, json=datav6)
    if 200 <= response.status_code < 300:
        return { _domain: sorted(response.json().get("measurements")) }, []
    else:
        raise Exception(f"Failed to create IPv6 measurement for {_domain}: {response.status_code} {response.text}")


import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

def main():
    parser = argparse.ArgumentParser(description="Process CDN origin for websites.")
    parser.add_argument("country", type=str, help="Country name to read and write JSON files")
    parser.add_argument('agg_func', type=str, help='Base longitude from VPN')
    args = parser.parse_args()

    probes_file = "probes.json"
    base_path = Path(f"/Users/joaomello/Desktop/tcc/results/{args.country}")
    input_file = base_path / f"output.json"

    latency_ipv4_output = base_path / "latency" / f"latency_ipv4.json"
    latency_ipv6_output = base_path / "latency" / f"latency_ipv6.json"
    icmp_block_output = base_path / "latency" / f"icmp_block_.json"
    fail_ipv6_output = base_path / "latency" / f"fail_ipv6_route.json"
    log_output = base_path / "latency" / f"log.json"
    raw_mesurements_output = base_path / "latency" / f"raw_mesurements.json"

    with open(input_file, "r") as f:
        websites = json.load(f)
    
    with open(probes_file, "r") as f:
        probes = json.load(f)

    mesurementsv4 = {}
    mesurementsv6 = {}
    total_fail_ipv6_route = []
    icmp_block = []
    log = ""

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {}
        for website in websites:
            futures[executor.submit(create_ping_measurement_v4, website, probes.get(args.country, {}).get("ases", {}))] = ("v4", website)
            futures[executor.submit(create_ping_measurement_v6, website, probes.get(args.country, {}).get("ases", {}))] = ("v6", website)
        
        for idx, future in enumerate(as_completed(futures)):
            proto, website = futures[future]
            try:
                if proto == "v4":
                    v4_result, v4_icmp_block = future.result()
                    mesurementsv4.update(v4_result)
                    icmp_block.extend(v4_icmp_block)
                elif proto == "v6":
                    v6_result, failed = future.result()
                    mesurementsv6.update(v6_result)
                    total_fail_ipv6_route.extend(failed)
            except Exception as e:
                log += f"Error processing {proto} {website}: {e}\n"
                print(e)
            print(f"Processed: {idx / (2 * len(websites)) * 100:.2f}%", end='\r')

    print(mesurementsv4)
    for i in range(10):
        print(f"Waiting... {10 * (i + 1)} seconds elapsed")
        time.sleep(10)

    ipv4_results = {}
    ipv6_results = {}
    counter = 1
    for website, mesurement_id in mesurementsv4.items():
        print(f"Processed: {counter / len(mesurementsv4.keys()) * 100:.2f}%", end='\r')
        counter += 1
        # Get the measurement ID
        api_key = "1cf1f929-9e31-4bae-ac1f-6487a82d8daa"
        result_url = "https://atlas.ripe.net/api/v2/measurements/{}/results?format=json"
        #for idx, mesurement in enumerate(mesurement_id):
        
        response = requests.get(result_url.format(mesurement_id[0])).json()
        log += f"Website: {website}\nMeasurement ID: {mesurement_id[0]}\nResponse: {response}\n\n\n\n"
        sorted_response_by_probe_id = sorted(response, key=lambda x: x.get("prb_id"))
        results_by_probe_map = map(lambda r: r.get("result"), sorted_response_by_probe_id)
        results_by_probe_flattened = map(lambda probe: [mesurement.get("rtt", None) for mesurement in probe], results_by_probe_map)
        ipv4_results.update({website: list(results_by_probe_flattened)})
    counter = 1
    for website, mesurement_id in mesurementsv6.items():
        print(f"Processed: {counter / len(mesurementsv6.keys()) * 100:.2f}%", end='\r')
        counter += 1
        # Get the measurement ID
        api_key = "1cf1f929-9e31-4bae-ac1f-6487a82d8daa"
        result_url = "https://atlas.ripe.net/api/v2/measurements/{}/results?format=json"
        #for idx, mesurement in enumerate(mesurement_id):
        
        response = requests.get(result_url.format(mesurement_id[0])).json()
        log += f"Website: {website}\nMeasurement ID: {mesurement_id[0]}\nResponse: {response}\n\n\n\n"
        sorted_response_by_probe_id = sorted(response, key=lambda x: x.get("prb_id"))
        results_by_probe_map = map(lambda r: r.get("result"), sorted_response_by_probe_id)
        results_by_probe_flattened = map(lambda probe: [mesurement.get("rtt", None) for mesurement in probe], results_by_probe_map)
        ipv6_results.update({website: list(results_by_probe_flattened)})
    
    

    # Save results
    with open(latency_ipv4_output, "w") as f:
        json.dump(ipv4_results, f, indent=4)

    with open(latency_ipv6_output, "w") as f:
        json.dump(ipv6_results, f, indent=4)
    
    with open(icmp_block_output, "w") as f:
        json.dump(icmp_block, f, indent=4)
    
    with open(fail_ipv6_output, "w") as f:
        json.dump(list(set(total_fail_ipv6_route)-set(icmp_block)), f, indent=4)
    
    with open(log_output, "w") as f:
        f.write(log)

    with open(raw_mesurements_output, "w") as f:
        raw = {}
        raw["v4"] = mesurementsv4
        raw["v6"] = mesurementsv6
        json.dump(raw, f, indent=4)


main()