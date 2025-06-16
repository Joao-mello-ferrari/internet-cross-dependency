import psycopg2
import ipaddress
import time
import subprocess
import threading
import tldextract
import socket

# Database connection
DB_NAME = "geo_db"
DB_USER = "postgres"
DB_PASS = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

def run_command(command):
    """Execute a shell command and return its output."""
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None

def run_script(args):
    start_time = time.time()
    print("Starting script", args[1])
    thread = threading.Thread(
        target=subprocess.run,
        args=(args,),
        kwargs={"check": True}
    )
    thread.start()
    thread.join()
    print(f"Time taken for {args[1]}: {time.time() - start_time} seconds \n")

def ip_to_int(ip):
    """Convert IP string to integer."""
    return int(ipaddress.IPv4Address(ip))

def get_country(ip):
    """Find the country for a given IP address."""
    try:
        ip_int = ip_to_int(ip)
    except Exception as e:
        return None
    
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()

    query = """SELECT find_location(%s);"""
    cur.execute(query, (ip_int,))
    result = cur.fetchone()
    
    cur.close()
    conn.close()
    
    return result[0].replace("(", "").replace(")", "").split(",") if result else None

def get_anycast(ip):
    """Find the country for a given IP address."""
    try:
        ip_int = ip_to_int(ip)
    except Exception as e:
        return None
    
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()
    
    query = """SELECT find_anycast(%s);"""

    cur.execute(query, (ip_int,))
    result = cur.fetchone()
    if result is None:
        return False
    
    return True if result[0] == "true" else False
    
def classify_provider(provider):
    if provider is None:
        return None
    
    # Define known cloud providers
    providers = ["amazon", "microsoft", "hostinger", "go-daddy", "azion", "cloudflarenet", "artfiles", "digitalocean", "akamai"]
    
    known_providers = {
        "amazon": "Amazon",
        "aws": "Amazon",
        "microsoft": "Microsoft",
        "azure": "Microsoft",
        "google": "Google",
        "gcp": "Google",
        "cloudflarenet": "Cloudflare",
        "cloudflare": "Cloudflare",
        "akamai": "Akamai",
        "oracle": "Oracle",
        "digitalocean": "DigitalOcean",
        "linode": "Linode",
        "hetzner": "Hetzner",
        "ovh": "OVH",
        "ibm": "IBM",
        "alibaba": "Alibaba",
        "taobao": "Alibaba",
        "artfiles": "Artfiles",
        "azion": "Azion",
        "go-daddy": "GoDaddy",
        "hostinger": "Hostinger",
    }
    for p in providers:
        if p in provider.lower():
            return known_providers.get(p)
    
        
    provider_lower = provider.lower().split(" ")[0]
    return known_providers.get(provider_lower, None)

def filter_unique_domains(urls, with_suffix=False):
    seen_domains = set()
    filtered_urls = []
    repeated_domains = {}

    for url in urls:
        # Extract the registered domain (e.g., google.com, example.com.br)
        extracted = tldextract.extract(url)
        main_domain = f"{extracted.domain}"
        if with_suffix:
            main_domain += f".{extracted.suffix}"

        if main_domain not in seen_domains:
            seen_domains.add(main_domain)
            filtered_urls.append(url)
        
        if main_domain in repeated_domains:
            repeated_domains[main_domain].append(url)
        else:
            repeated_domains[main_domain] = [url]
    
    keys_to_remove = [key for key in repeated_domains if len(repeated_domains[key]) == 1]
    for key in keys_to_remove:
        repeated_domains.pop(key)

    return filtered_urls, repeated_domains

import re
def get_ipv6_from_dns(hostname):
    try:
        # Perform DNS query for the hostname
        results = run_command(["dig", "+short", hostname, "AAAA"])
        # Use regex to extract the first IPv6 address from the results
        ipv6_pattern = r"([0-9a-fA-F]{1,4}(:[0-9a-fA-F]{1,4}){2,7})"
        matches = re.search(ipv6_pattern, results)
        return matches.group(1) if matches else None
    except Exception:
        return None

#print(get_ipv6_from_ip("216.238.108.120"))  # Google's public DNS




import math

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two lat/lon points (km)."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c  # Distance in km

def latency_match_geolocation(lat1, lon1, lat2, lon2, latency_ms, medium="fiber"):
    """
    Check if the given latency is physically possible for the distance.

    Parameters:
    lat1, lon1 - Coordinates of point 1 (degrees)
    lat2, lon2 - Coordinates of point 2 (degrees)
    latency_ms - Measured latency (milliseconds, round-trip)
    medium - "fiber" (200,000 km/s) or "air" (300,000 km/s)

    Returns:
    True if the latency is realistic, False otherwise
    """
    speed = 200_000 if medium == "fiber" else 300_000  # km/s
    distance = haversine(lat1, lon1, lat2, lon2)

    max_possible_distance = (latency_ms / 1000) * (speed / 2)  # One-way distance
    lowered_distance = max_possible_distance * 0.8  # 20% less distance for latency

    return 0.5 < distance / lowered_distance < 2

# Example Usage:
lat1, lon1 = 48.8566, 2.3522   # Paris
lat2, lon2 = 40.7128, -74.0060 # New York
latency_ms = 70  # Measured round-trip latency in milliseconds

#print(is_latency_physical(lat1, lon1, lat2, lon2, latency_ms, medium="fiber"))
#print(get_country("13.227.126.21").replace("(", "").replace(")", "").split(","))
#print(latency_match_geolocation(-32, -52, 40, -74, 200))
