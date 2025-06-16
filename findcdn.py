import argparse
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from helpers import run_command

IPV4_REGEX = re.compile(r"\d+\.\d+\.\d+\.\d+")

def extract_ip(text):
    """Extract the first IPv4 address from a given text."""
    match = IPV4_REGEX.search(text)
    return match.group() if match else None


def process_website(website):
    """Process a single website to determine its CDN and locality."""
    site = website.replace("http://", "").replace("https://", "")
    cdn_result, ip_result = {}, {}
    
    try:
        cdn_output = run_command(["findcdn", "list", site])
    
        if not cdn_output: # Findcdn returned an error
            return {}, {}
        
        # Process findcdn output
        cdn_info = json.loads(re.search(r'\{(?:[^{}]++|\{(?:[^{}]++|\{[^{}]*\})*\})*\}', cdn_output, re.DOTALL).group())
        website_result = cdn_info["domains"]
        
        if website_result != {} and website_result:
            # Findcdn returned not empty results
            cdn_result[site] = website_result[site]["cdns_by_names"]
            ip_result[site] = extract_ip(website_result[site]["IP"])
        else:
            cdn_result[site] = None
            ip_result[site] = None
    
    except Exception as e:
        print(f"Error on process_website ({site}): {e}")
    
    return cdn_result, ip_result

def main():
    parser = argparse.ArgumentParser(description="Process CDN origin for websites.")
    parser.add_argument("country", type=str, help="Country name to read and write JSON files")
    parser.add_argument("vpn", type=str, help="Country name to read and write JSON files")
    args = parser.parse_args()

    base_path = Path(f"/Users/joaomello/Desktop/tcc/results/{args.country}")
    input_file = base_path / f"output.json"
    cdn_output = base_path / "locality" / args.vpn / f"cdn.json"

    with open(input_file, "r") as f:
        websites = json.load(f)

    cdn_results, ips_results = {}, {}

    # Process websites concurrently
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = {executor.submit(process_website, website): website for website in websites}

        # Process results as they complete
        for idx, future in enumerate(as_completed(futures)):
            print(f"Processed: {(idx + 1) / len(websites) * 100:.2f}%", end='\r')
            result, location_result = future.result()
            cdn_results.update(result)
            ips_results.update(location_result)

    # Save results
    with open(cdn_output, "w") as f:
        json.dump(cdn_results, f, indent=4)

if __name__ == "__main__":
    main()
