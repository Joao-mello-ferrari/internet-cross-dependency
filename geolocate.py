import argparse
import json
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from helpers import run_command, classify_provider, get_country, get_anycast, latency_match_geolocation
from ipwhois import IPWhois
from statistics import median, mean

agg_func_map = {
    "min": min,
    "mean": mean,
    "median": median,
}

def get_provider(ip, site):
    try:
        obj = IPWhois(ip)
        results = obj.lookup_whois()  # Using RDAP instead of WHOIS for better accuracy
        asn_description = results.get("asn_description")
        return classify_provider(asn_description.split(",")[0])
    except Exception as e:
        #print(f"Error on get_provider ({ip}, {site}))): {e}")
        return None
 
def dig(site):
    try:
        return re.search(r'\d+\.\d+\.\d+\.\d+', run_command(["dig", "+short", site, "@8.8.8.8"])).group()
    except Exception as e: 
        print(f"Error on dig ({site}))): {e}")
        return None

def process_website(website, country, agg_func):
    """Process a single website to determine its CDN and locality."""
    site = website.replace("http://", "").replace("https://", "")
    location_result, provider_result, ip_result, websites_with_error = {}, {}, {}, []
    
    try:
        # Extract IP address from google DNS server
        ip_address = dig(site)

        # Set the ip to dig returned ip
        ip_result[site] = ip_address

        # Search for the provider of the IP address, since findCdn could not find it
        provider = get_provider(ip_address, site)
        provider_result[site] = provider
        
        # Append error results to retry later
        if provider is None:
            websites_with_error.append(site)
    
        # Get the country of the IP address, using ipinfo.io database
        serve_country, _, _ = get_country(ip_address)
        any_cast = ""
        if get_anycast(ip_address):
            any_cast = " - anycast"
        
        if serve_country is None:
            location_result[site] = None
        else:
            location_result[site] = serve_country.lower() + any_cast
        # elif serve_country.lower() == country:
        #     location_result[site] = "Local" + any_cast
        # else:
        #     location_result[site] = "External" + any_cast

    except Exception as e:
        print(f"Error on process_website ({site}): {e}")
    
    return location_result, provider_result, ip_result

def main():
    parser = argparse.ArgumentParser(description="Process CDN origin for websites.")
    parser.add_argument("country", type=str, help="Country name to read and write JSON files")
    parser.add_argument("vpn", type=str, help="Country name to read and write JSON files")
    parser.add_argument('agg_func', type=str, help='Base longitude from VPN')
    args = parser.parse_args()

    base_path = Path(f"/Users/joaomello/Desktop/tcc/results/{args.country}")
    input_file = base_path / f"output.json"
    location_output_file = base_path / "locality" / args.vpn / f"location.json"
    provider_output_file = base_path / "locality" / args.vpn / f"provider.json"
    ip_ouput_file = base_path / "locality" / args.vpn / f"ips.json"

    with open(input_file, "r") as f:
        websites = json.load(f)

    location_results, provider_results, ips_results = {}, {}, {}

    # Process websites concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_website, website, args.country, args.agg_func): website for website in websites}
        for idx, future in enumerate(as_completed(futures)):
            print(f"Processed: {idx / len(websites) * 100:.2f}%", end='\r')
            
            location_result, provider_result, ip_result = future.result()
            location_results.update(location_result)
            provider_results.update(provider_result)
            ips_results.update(ip_result)
    

    # Save results
    with open(location_output_file, "w") as f:
        json.dump(location_results, f, indent=4)

    with open(provider_output_file, "w") as f:
        json.dump(provider_results, f, indent=4)
    
    with open(ip_ouput_file, "w") as f:
        json.dump(ips_results, f, indent=4)

if __name__ == "__main__":
    main()
