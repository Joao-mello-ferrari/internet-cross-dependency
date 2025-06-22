import argparse
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from lib import run_command

# ===================
# Regex and Constants
# ===================
IPV4_REGEX = re.compile(r"\d+\.\d+\.\d+\.\d+")
FINDCDN_JSON_REGEX = re.compile(
    r'\{(?:[^{}]++|\{(?:[^{}]++|\{[^{}]*\})*\})*\}',
    re.DOTALL
)


# ===================
# Helper Functions
# ===================
def extract_ip(text):
    """Extract the first IPv4 address from a given text."""
    match = IPV4_REGEX.search(text)
    return match.group() if match else None


def process_website(website):
    """Process a single website to determine its CDN and IP."""
    site = website.replace("http://", "").replace("https://", "")
    cdn_result, ip_result = {}, {}

    try:
        output = run_command(["findcdn", "list", site])
        if not output:
            return {}, {}

        match = FINDCDN_JSON_REGEX.search(output)
        if not match:
            return {}, {}

        cdn_info = json.loads(match.group())
        site_info = cdn_info.get("domains", {}).get(site, {})

        cdn_result[site] = site_info.get("cdns_by_names")
        ip_result[site] = extract_ip(site_info.get("IP", ""))

    except Exception as e:
        print(f"[❌] Error processing {site}: {e}")

    return cdn_result, ip_result


def save_json(data, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


# ===================
# Main Logic
# ===================
def main():
    parser = argparse.ArgumentParser(description="Process CDN and IP info for websites.")
    parser.add_argument("--country", type=str.lower, required=True, help="Country code (label)")
    parser.add_argument("--code", type=str.lower, required=True, help="Country code (folder)")
    parser.add_argument("--vpn", type=str.lower, required=True, help="VPN country code (locality folder)")
    args = parser.parse_args()

    # Paths
    base_path = Path(f"results/{args.code}")
    input_file = base_path / "output.json"
    output_file = base_path / "locality" / args.vpn / "cdn.json"

    # Load websites
    with open(input_file) as f:
        websites = json.load(f)

    cdn_results, ip_results = {}, {}

    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(process_website, site): site for site in websites}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing websites"):
            result, ip = future.result()
            cdn_results.update(result)
            ip_results.update(ip)

    # Save
    save_json(cdn_results, output_file)

    print(f"\n✅ CDN data saved to {output_file}")


if __name__ == "__main__":
    main()
