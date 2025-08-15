import json
import argparse
import os
import httpx
from pathlib import Path
from openai import OpenAI
from math import ceil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from os import getenv
from dotenv import load_dotenv
from src.steps.website_classification.helpers import classify_batch, narrow_classes


# =======================
# OPENAI CLIENT
# =======================
load_dotenv()
client = OpenAI(
    api_key=getenv("OPENAI_API_KEY"),
    http_client=httpx.Client(verify=False)
)

# =======================
# CLI ARGUMENTS
# =======================
parser = argparse.ArgumentParser(description='Classify website categories.')
parser.add_argument('--country', type=str, required=True, help='Country name to read and write JSON files')
parser.add_argument('--code', type=str.lower, required=True, help='Country code (e.g. BR, US) for file paths')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size (default: 10)')
parser.add_argument('--workers', type=int, default=10, help='Number of concurrent threads (default: 10)')
args = parser.parse_args()

# =======================
# LOAD INPUT DATA
# =======================
websites_path = Path(f"results/{args.code}/output.json")
with open(websites_path, 'r') as f:
    websites = json.load(f)
    websites = list(set(websites))


# =======================
# GET CLASSIFICATION RESULTS FROM CACHE
# =======================
classes_cache_path = Path(f"results/classified_websites.json")
website_to_process = []
with open(classes_cache_path, 'r') as f:
    cached_results = json.load(f)
    cached_results_keys = list(cached_results.keys())

    for site in websites:
        if site not in cached_results_keys:
            website_to_process.append(site)

# =======================
# PREPARE BATCHES
# =======================
batch_size = args.batch_size
batches = [
    website_to_process[i * batch_size: (i + 1) * batch_size]
    for i in range(ceil(len(website_to_process) / batch_size))
]


# =======================
# PROCESS IN PARALLEL
# =======================
results = {}
with ThreadPoolExecutor(max_workers=args.workers) as executor:
    future_to_batch = {executor.submit(classify_batch, batch, client): batch for batch in batches}

    for future in tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc="Processing"):
        batch_result = future.result()
        if batch_result:
            results.update(batch_result)


# =======================
# SAVE OUTPUTS
# =======================
# Save raw results before narrowing classes
raw_output_path = Path(f"results/classified_websites_raw.json")
with open(raw_output_path, 'r') as f:
    raw_cached_results = json.load(f)

raw_cached_results.update(results)
with open(raw_output_path, 'w') as f:
    json.dump(raw_cached_results, f, indent=2)

# Save raw results before narrowing classes
output_path = Path(f"results/classified_websites.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    results = narrow_classes(results)
    cached_results.update(results)
    json.dump(cached_results, f, indent=2)

print(f"\n✅ Classification complete for {len(results.keys())} new websites")
