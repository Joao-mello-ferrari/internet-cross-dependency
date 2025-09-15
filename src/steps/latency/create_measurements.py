import os
import time
import requests
from ping3 import ping

from src.steps.latency.helpers import find_ipv6_has_address

RIPE_ATLAS_MEASUREMENTS_URL = "https://atlas.ripe.net/api/v2/measurements/"

def get_probes_list(country_ases, af):
    key = "v4" if af == 4 else "v6"
    probes = []
    for country_as in country_ases.values():
        probes.extend(country_as.get(key, []))
    return probes

def create_ping_measurement(domain, country_ases, af, attempts=3):
    domain = domain.replace("http://", "").replace("https://", "")
    accepts_icmp = ping(domain, timeout=1)
    if not accepts_icmp:
        return {}, domain, None
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
            "packets": 5,
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
        "start_time": int(time.time()) + 10,
    }

    try:
        resp = requests.post(RIPE_ATLAS_MEASUREMENTS_URL, headers=headers, json=payload, timeout=5)
        if resp.ok:
            return {domain: resp.json().get("measurements")}, None, None
        else:
            raise Exception(f"Failed to create AF{af} measurement for {domain}: {resp.status_code} {resp.text}")
    except requests.exceptions.Timeout:
        print("Request timed out! Retrying...")
        if attempts > 0:
            time.sleep(2*(4-attempts))
            return create_ping_measurement(domain, country_ases, af, attempts - 1)
        raise Exception("Request timed out while creating measurement.")