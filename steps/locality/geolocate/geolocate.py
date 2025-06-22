import argparse
import json
import re
from pathlib import Path
from ipwhois import IPWhois
from statistics import median, mean
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from steps.locality.geolocate.helpers import classify_provider, get_country, get_anycast
from lib import run_command


# ===================
# Constants and Regex
# ===================
IPV4_REGEX = re.compile(r"\d+\.\d+\.\d+\.\d+")

AGG_FUNC_MAP = {
    "min": min,
    "mean": mean,
    "median": median,
}


# ===================
# Helper Functions
# ===================
def dig(site):
    """Perform DNS lookup to extract an IPv4 address."""
    try:
        output = run_command(["dig", "+short", site, "@8.8.8.8"])
        match = IPV4_REGEX.search(output)
        return match.group() if match else None
    except Exception as e:
        print(f"[❌] Error in dig({site}): {e}")
        return None


def get_provider(ip):
    """Get the provider name based on the IP's WHOIS info."""
    try:
        obj = IPWhois(ip)
        results = obj.lookup_whois()
        asn_description = results.get("asn_description")
        if asn_description:
            return classify_provider(asn_description.split(",")[0])
    except Exception:
        pass
    return None


def process_website(website, country):
    """Process a single website to get location, provider, and IP info."""
    site = website.replace("http://", "").replace("https://", "")
    location_result, provider_result, ip_result = {}, {}, {}

    try:
        ip_address = dig(site)
        ip_result[site] = ip_address

        if not ip_address:
            location_result[site] = None
            provider_result[site] = None
            return location_result, provider_result, ip_result

        provider = get_provider(ip_address)
        provider_result[site] = provider

        serve_country, _, _ = get_country(ip_address)
        anycast_tag = " - anycast" if get_anycast(ip_address) else ""

        if serve_country:
            location_result[site] = serve_country.lower() + anycast_tag
        else:
            location_result[site] = None

    except Exception as e:
        print(f"[❌] Error processing {site}: {e}")

    return location_result, provider_result, ip_result


# ===================
# Main
# ===================
def main():
    parser = argparse.ArgumentParser(description="Geolocate CDN and IPs for websites.")
    parser.add_argument("--country", required=True, type=str.lower, help="Country code")
    parser.add_argument("--code", required=True, type=str.lower, help="Country code")
    parser.add_argument("--vpn", required=True, type=str, help="VPN location name")
    args = parser.parse_args()

    # === Paths ===
    base_path = Path(f"results/{args.code}")
    input_file = base_path / "output.json"
    output_dir = base_path / "locality" / args.vpn
    output_dir.mkdir(parents=True, exist_ok=True)

    location_output_file = output_dir / "location.json"
    provider_output_file = output_dir / "provider.json"
    ip_output_file = output_dir / "ips.json"

    # === Load Websites ===
    with open(input_file, "r") as f:
        websites = json.load(f)

    location_results, provider_results, ips_results = {}, {}, {}

    # === Process Websites ===
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(process_website, website, args.code): website
            for website in websites
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing websites"):
            location_result, provider_result, ip_result = future.result()
            location_results.update(location_result)
            provider_results.update(provider_result)
            ips_results.update(ip_result)

    # === Save Outputs ===
    with open(location_output_file, "w") as f:
        json.dump(location_results, f, indent=4)

    with open(provider_output_file, "w") as f:
        json.dump(provider_results, f, indent=4)

    with open(ip_output_file, "w") as f:
        json.dump(ips_results, f, indent=4)

    print(f"✅ Results saved in {output_dir}")


if __name__ == "__main__":
    main()
