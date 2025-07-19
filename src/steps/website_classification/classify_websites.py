import json
import argparse
import os
import httpx
from openai import OpenAI
from math import ceil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from os import getenv
from dotenv import load_dotenv
from src.steps.website_classification.helpers import classify_batch


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
parser.add_argument('input', type=str, help='Input JSON file with list of websites')
parser.add_argument('output', type=str, help='Output JSON file for results')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size (default: 10)')
parser.add_argument('--workers', type=int, default=10, help='Number of concurrent threads (default: 10)')
args = parser.parse_args()


# =======================
# LOAD INPUT DATA
# =======================
with open(args.input, 'r') as f:
    websites = json.load(f)
websites = list(set(websites))


# =======================
# PREPARE BATCHES
# =======================
batch_size = args.batch_size
batches = [
    websites[i * batch_size: (i + 1) * batch_size]
    for i in range(ceil(len(websites) / batch_size))
]


# =======================
# PROCESS IN PARALLEL
# =======================
results = {}

with ThreadPoolExecutor(max_workers=args.workers) as executor:
    future_to_batch = {executor.submit(classify_batch, batch): batch for batch in batches}

    for future in tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc="Processing"):
        batch_result = future.result()
        if batch_result:
            results.update(batch_result)


# =======================
# SAVE OUTPUT
# =======================
os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Classification complete. Saved {len(results)} results to {args.output}")
