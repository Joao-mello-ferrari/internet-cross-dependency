import argparse
import requests
import json
import os
from pathlib import Path
from itertools import combinations

from src.steps.latency.helpers import country_name_mapper, haversine_distance

ASPOP_URL = "https://stats.labs.apnic.net/cgi-bin/aspop?c={}&d=09/06/2025&f=j"
RIPE_ATLAS_PROBES_URL = "https://atlas.ripe.net/api/v2/probes/"
MIN_PROBES_PER_PROTOCOL = 3
MAX_PROBES_PER_PROTOCOL = 12

def get_probes_without_asn(country_code):
    """
    Fetch RIPE Atlas probes in the given country that do not have an ASN assigned
    """
    params = {
        'country_code__in': country_code,
        'status': 1,  # Only connected probes
        'page_size': 100  # Get a good number of probes to choose from
    }
    
    response = requests.get(RIPE_ATLAS_PROBES_URL, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch probes without ASN in {country_code}: {response.status_code}")
        return []
    
    probes_data = response.json()
    return probes_data.get('results', [])

def get_probes_for_as(asn, country_code):
    """
    Search for RIPE Atlas probes that include the AS number
    and have a country code matching the given country
    """
    
    # Remove 'AS' prefix if present
    if isinstance(asn, str) and asn.startswith('AS'):
        asn = asn[2:]
    
    params = {
        'asn_v4': asn,
        'country_code': country_code,
        'status': 1,  # Only connected probes
        'page_size': 100  # Get a good number of probes to choose from
    }
    
    response = requests.get(RIPE_ATLAS_PROBES_URL, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch probes for AS{asn} in {country_code}: {response.status_code}")
        return []
    
    probes_data = response.json()
    return probes_data.get('results', [])


def select_distant_probes(probes, max_probes=4, ipv6_boost_factor=2.0):
    """
    Highlight probes that are as far away as possible from each other (use lat and lon)
    returning up to 4 probes per AS, with preference for IPv6-capable probes
    """
    if not probes:
        return []
        
    if len(probes) <= max_probes:
        return probes
    
    # Filter out probes without coordinates
    valid_probes = [p for p in probes if p.get('geometry') and p['geometry'].get('coordinates')]
    
    if len(valid_probes) <= max_probes:
        return valid_probes
    
    # Calculate distance between each pair of probes
    distances = {}
    for p1, p2 in combinations(valid_probes, 2):
        try:
            p1_lon, p1_lat = p1['geometry']['coordinates']
            p2_lon, p2_lat = p2['geometry']['coordinates']
            
            dist = haversine_distance(p1_lat, p1_lon, p2_lat, p2_lon)
            # Create consistent key using min/max
            pair_key = (min(p1['id'], p2['id']), max(p1['id'], p2['id']))
            distances[pair_key] = dist
        except (TypeError, ValueError):
            continue
    
    # Greedy algorithm to select probes that are far from each other
    selected_probes = []
    remaining_probes = valid_probes.copy()
    
    # Start with an IPv6-capable probe if available
    ipv6_probes = [p for p in remaining_probes if p.get('asn_v6')]
    if ipv6_probes:
        selected_probes.append(ipv6_probes[0])
        remaining_probes.remove(ipv6_probes[0])
    elif remaining_probes:
        selected_probes.append(remaining_probes[0])
        remaining_probes.remove(remaining_probes[0])
    
    # Select the rest of the probes based on maximum minimum distance
    # with a boost for IPv6-capable probes
    while len(selected_probes) < max_probes and remaining_probes:
        max_adjusted_dist = -1
        next_probe = None
        
        for probe in remaining_probes:
            min_dist = float('inf')
            for selected in selected_probes:
                pair_key = (min(probe['id'], selected['id']), max(probe['id'], selected['id']))
                if pair_key in distances:
                    min_dist = min(min_dist, distances[pair_key])
                else:
                    min_dist = 0  # If we can't calculate distance, assume they're close
            
            # Apply boost factor for IPv6-capable probes
            adjusted_dist = min_dist
            if probe.get('asn_v6'):
                adjusted_dist *= ipv6_boost_factor
            
            if adjusted_dist > max_adjusted_dist:
                max_adjusted_dist = adjusted_dist
                next_probe = probe
        
        if next_probe:
            selected_probes.append(next_probe)
            remaining_probes.remove(next_probe)
        else:
            break
    
    return selected_probes


def extract_probe_info(probe):
    """
    Extract relevant information from a probe object
    """
    return {
        'id': probe.get('id'),
        'asn_v4': probe.get('asn_v4'),
        'asn_v6': probe.get('asn_v6'),
        #'coordinates': probe.get('geometry', {}).get('coordinates', []),
        #'address_v4': probe.get('address_v4'),
        #'address_v6': probe.get('address_v6'),
        #'country_code': probe.get('country_code'),
        #'is_public': probe.get('is_public'),
        #'status': probe.get('status')
    }

def weighted_clip_probes(ases_dict, protocol_key, max_total):
        """
        Clip probes using weighted distribution based on current probe count per AS
        """
        # Calculate current totals and weights
        as_probe_counts = {}
        total_current = 0
        
        for as_num, as_data in ases_dict.items():
            if as_num == "default":  # Skip default for now
                continue
            probe_count = len(as_data.get(protocol_key, []))
            as_probe_counts[as_num] = probe_count
            total_current += probe_count
        
        if total_current <= max_total:
            return  # No clipping needed
        
        # Calculate weights (proportion of probes each AS should keep)
        total_weight = sum(as_probe_counts.values())
        if total_weight == 0:
            return
        
        # Distribute probes proportionally
        remaining_probes = max_total
        for as_num in sorted(as_probe_counts.keys()):  # Sort for consistent results
            current_probes = ases_dict[as_num][protocol_key]
            if remaining_probes <= 0:
                # If no probes left, clear this AS
                ases_dict[as_num][protocol_key] = current_probes[:1]
                continue
                
            current_count = as_probe_counts[as_num]
            if current_count == 0:
                continue
                
            # Calculate how many probes this AS should get
            weight = current_count / total_weight
            allocated_probes = max(1, min(remaining_probes, round(weight * max_total)))
            
            # Clip the probes list for this AS
            ases_dict[as_num][protocol_key] = current_probes[:allocated_probes]
            remaining_probes -= allocated_probes
    


def fetch_probes(country_code):
    aspop_response = requests.get(ASPOP_URL.format(country_code))
    if aspop_response.status_code != 200:
        raise Exception(f"Failed to fetch AS-POP data for {country_code}: {aspop_response.status_code} {aspop_response.text}")

    # Filter AS-POP data to include only those with more than 2% of CC population
    aspop_data = list(filter(lambda _as: _as.get("Percent of CC Pop", 0) > 2.0, aspop_response.json()["Data"]))
    
    # Implementation of the functionality described in the comments
    result = {}
    total_percent = 0
    v4_probes_counter = 0
    v6_probes_counter = 0
    for as_info in aspop_data:
        as_number = as_info.get('AS')
        if as_number:
            # Get probes for this AS and country
            country_code = country_code.upper()
            probes = get_probes_for_as(as_number, country_code)
            
            # Select up to 8 probes that are as far away from each other as possible
            selected_probes = select_distant_probes(probes)
            
            # Extract relevant probe information
            #probe_info = [extract_probe_info(probe) for probe in selected_probes]
            ipv4_probes = [probe.get("id") for probe in selected_probes if probe.get("asn_v4")]
            ipv6_probes = [probe.get("id") for probe in selected_probes if probe.get("asn_v6")]
            
            if not ipv4_probes and not ipv6_probes:
              print(f"Skipping AS{as_number} since no eligible probes were found")
              continue

            # Add to result
            result[as_number] = {
                'v4': ipv4_probes,
                'v6': ipv6_probes,
                'percent': as_info.get('Percent of CC Pop', 0),
            }
            total_percent += as_info.get('Percent of CC Pop', 0)
            v4_probes_counter += len(ipv4_probes)
            v6_probes_counter += len(ipv6_probes)
    
    
    # Apply weighted clipping
    weighted_clip_probes(result, 'v4', MAX_PROBES_PER_PROTOCOL)
    weighted_clip_probes(result, 'v6', MAX_PROBES_PER_PROTOCOL)
    
    # Recalculate counters after clipping
    v4_probes_counter = sum(len(as_data.get('v4', [])) for as_data in result.values() if isinstance(as_data, dict))
    v6_probes_counter = sum(len(as_data.get('v6', [])) for as_data in result.values() if isinstance(as_data, dict))

    # Ensure at least MIN_PROBES_PER_COUNTRY probes for each protocol
    # by adding probes without ASN if necessary
    if v4_probes_counter < MIN_PROBES_PER_PROTOCOL or v6_probes_counter < MIN_PROBES_PER_PROTOCOL:
        default_probes = get_probes_without_asn(country_code=country_code)

        # Calculate how many more probes we need, but don't exceed MAX_PROBES_PER_PROTOCOL
        remaining_v4 = max(0, min(MIN_PROBES_PER_PROTOCOL - v4_probes_counter, 
                                 MAX_PROBES_PER_PROTOCOL - v4_probes_counter))
        remaining_v6 = max(0, min(MIN_PROBES_PER_PROTOCOL - v6_probes_counter,
                                 MAX_PROBES_PER_PROTOCOL - v6_probes_counter))

        v4_probes = []
        v6_probes = []
        
        for probe in default_probes:
            if remaining_v4 <= 0 and remaining_v6 <= 0:
                break
            
            if probe.get('asn_v4') and remaining_v4 > 0:
                v4_probes.append(probe.get('id'))
                remaining_v4 -= 1
            
            if probe.get('asn_v6') and remaining_v6 > 0:
                v6_probes.append(probe.get('id'))
                remaining_v6 -= 1
        
        if v4_probes or v6_probes:
            result["default"] = {
                'v4': v4_probes,
                'v6': v6_probes,
            }


    country_probes = {
      'name': country_name_mapper.get(country_code, country_code),
      'ases': result,
      'percent': total_percent,
    }

    # Save the result to a JSON file
    output_file = Path(f"results/{country_code}/probes.json")
    os.makedirs(output_file.parent, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(country_probes, f, indent=2)
    
    print(f"Saved {len(result)} AS entries with probes data to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Fetch RIPE Atlas probes for a given country.")
    parser.add_argument("--country", type=str.lower, required=True, help="Country code (e.g., BR, US) to process probes data")
    
    args = parser.parse_args()
    
    fetch_probes(args.country)

if __name__ == "__main__":
    main()