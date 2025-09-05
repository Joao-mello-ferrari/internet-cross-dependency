import numpy as np
from json import load
from statistics import median
from pathlib import Path
from itertools import zip_longest


def aggregate_ping_by_domain(domain_probes_v4=[], domain_probes_v6=[], require_both_procols_results=True):
    ipv4_results, ipv6_results = [], []

    for probe_v4, probe_v6 in zip_longest(domain_probes_v4, domain_probes_v6, fillvalue=[]):
        valid_v4 = list(filter(lambda x: x not in (None, 0), probe_v4))
        valid_v6 = list(filter(lambda x: x not in (None, 0), probe_v6))

        if require_both_procols_results and (not valid_v4 or not valid_v6):
            continue

        if valid_v4:
            ipv4_results.append(median(valid_v4))
        if valid_v6:
            ipv6_results.append(median(valid_v6))

    return (
        [median(ipv4_results)] if ipv4_results else [],
        [median(ipv6_results)] if ipv6_results else []
    )

def aggregate_ping_by_probes(domain_probes_v4=[], domain_probes_v6=[], require_both_procols_results=True):
    ipv4_vals, ipv6_vals = [], []
    ipv4_err, ipv6_err = 0, 0

    for probe_v4, probe_v6 in zip_longest(domain_probes_v4, domain_probes_v6, fillvalue=[]):
        valid_v4 = list(filter(lambda x: x not in (None, 0), probe_v4))
        valid_v6 = list(filter(lambda x: x not in (None, 0), probe_v6))

        if require_both_procols_results and (not valid_v4 or not valid_v6):
            continue

        if not valid_v4:
            ipv4_err += 1
        else:
            ipv4_vals.append(median(valid_v4))

        if not valid_v6:
            ipv6_err += 1
        else:
            ipv6_vals.append(median(valid_v6))

    return ipv4_vals, ipv6_vals, ipv4_err, ipv6_err


def setup_paths(code):
    # --- Paths Setup ---
    base_path = Path(f"/Users/joaomello/Desktop/tcc/results/{code}")
    (base_path / "results" / "latency").mkdir(parents=True, exist_ok=True)
    (base_path / "results" / "latency" / "by_domain").mkdir(parents=True, exist_ok=True)
    (base_path / "results" / "latency" / "by_probe").mkdir(parents=True, exist_ok=True)

    latency_ipv4_input = base_path / "latency" / "latency_ipv4.json"
    latency_ipv6_input = base_path / "latency" / "latency_ipv6.json"
    fail_ipv6_input = base_path / "latency" / "fail_ipv6_route.json"

    return base_path, latency_ipv4_input, latency_ipv6_input, fail_ipv6_input


def get_rtts(ipv4_input, ipv6_input, fail_ipv6_input, require_both_procols_results=True):
  # --- Load and Process Latency Data ---
  with open(ipv4_input) as f_v4, open(ipv6_input) as f_v6:
    data_v4 = load(f_v4)
    data_v6 = load(f_v6)

    domains = set(data_v4) | set(data_v6) # Join the domains from both protocols
    ipv4_by_domain, ipv6_by_domain = [], []
    ipv4_by_probe, ipv6_by_probe = [], []

    for domain in domains:
        res_v4_domain, res_v6_domain = aggregate_ping_by_domain(
            data_v4.get(domain, []), data_v6.get(domain, []), require_both_procols_results
        )
        res_v4_probe, res_v6_probe, _, _ = aggregate_ping_by_probes(
            data_v4.get(domain, []), data_v6.get(domain, []), require_both_procols_results
        )

        ipv4_by_domain += res_v4_domain
        ipv6_by_domain += res_v6_domain
        ipv4_by_probe += res_v4_probe
        ipv6_by_probe += res_v6_probe

    ipv4_by_domain = np.array(ipv4_by_domain)
    ipv6_by_domain = np.array(ipv6_by_domain)
    ipv4_by_probe = np.array(ipv4_by_probe)
    ipv6_by_probe = np.array(ipv6_by_probe)

  # --- Load Failures (not used directly) ---
  with open(fail_ipv6_input) as f:
      fail_ipv6_route = load(f)
  
  return ipv4_by_domain, ipv6_by_domain, ipv4_by_probe, ipv6_by_probe, fail_ipv6_route, len(domains)