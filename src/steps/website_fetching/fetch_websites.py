import json
import argparse
import os
from math import ceil
from google.cloud import bigquery
from src.steps.website_fetching.queries import country, get_global
from src.steps.website_fetching.helpers import filter_unique_domains
from tqdm import tqdm

# ==========================
# Query Map
# ==========================
queries_map = {
    "country": country,
    "global": get_global
}

# ==========================
# CLI Arguments
# ==========================
parser = argparse.ArgumentParser(description='Process latency for websites.')

parser.add_argument('--country', type=str, required=True, help='Country name to read and write JSON files')
parser.add_argument('--code', type=str.lower, required=True, help='Country code (e.g. BR, US) for file paths')
parser.add_argument('--query', type=str, required=True, choices=queries_map.keys(), help='Type of query: "country" or "global"')
parser.add_argument('--semester', type=str, required=True, help='Semester to fetch websites from (e.g. 2024-2)')
parser.add_argument('--amount', type=int, required=True, help='Number of websites to fetch')
parser.add_argument('--filter-dns', action='store_true', help='Apply DNS deduplication for base domain names')

args = parser.parse_args()

# ==========================
# Query Selection
# ==========================
query = queries_map.get(args.query)
if query is None:
    print("❌ Invalid query")
    exit(1)

# ==========================
# Query Execution
# ==========================
results = []
pool_size = min(100, args.amount)
range_count = ceil(args.amount / 100)
client = bigquery.Client(project="crux-466413")

for i in tqdm(range(range_count), desc="Fetching websites", unit="batch"):
    sql_query = query(
        country_code=args.code,
        offset=i * pool_size,
        limit=pool_size,
        semester=args.semester
    )

    try:
        query_job = client.query(sql_query)
        rows = query_job.result()
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        exit(1)

    data = list(map(lambda row: row.origin, rows))
    results.extend(data)

# ==========================
# Output Handling
# ==========================
output_dir = os.path.join("results", args.code)
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "output.json")

# Save initial output
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Processed {len(results)} websites for {args.code}")

# DNS Filter Option
if args.filter_dns:
    filtered, repeated = filter_unique_domains(results, False)

    repeated_path = os.path.join(output_dir, "repeated_output.json")
    with open(repeated_path, "w") as f:
        json.dump(repeated, f, indent=2)

    print(f"🔍 DNS filter applied. Removed {len(results) - len(filtered)} duplicated base domain names.")

print(f"✅ Output saved to {output_path}")