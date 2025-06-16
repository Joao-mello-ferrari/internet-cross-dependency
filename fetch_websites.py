import json
import argparse
import os
from queries import country, get_global
from helpers import run_command, filter_unique_domains
from math import ceil

queriesMap = {
    "country": country,
    "global": get_global
}

parser = argparse.ArgumentParser(description='Process latency for websites.')
parser.add_argument('country', type=str, help='Country name to read and write JSON files')
parser.add_argument('code', type=str, help='Country code to read and write JSON files')
parser.add_argument('query', type=str, help='Type of query')
parser.add_argument('semester', type=str, help='Semester to fetch websites from')
parser.add_argument('amount', type=str, help='Amount of websites')
parser.add_argument('filter_dns', type=str, help='Filter repeated dns')
args = parser.parse_args()

query = queriesMap.get(args.query)
if query is None:    
    print("Invalid query")
    exit()

results = []
poolSize = min(100, int(args.amount))
_range = ceil(int(args.amount) / 100)
for i in range(_range):
    by_country = query(country_code=args.code, offset=i * poolSize, limit=poolSize, semester=args.semester)
    result = run_command(["bq", "query", "--use_legacy_sql=false", "--format=json", by_country])

    if result is None:
        print("Query failed")
        exit()  

    data = json.loads(result)
    results.extend(data)
    
    print(f"Processed: {i / _range * 100:.2f}%", end='\r')

with open(f"results/{args.code}/output.json", "w") as f:
    results = list(map(lambda result: result.get("origin"), results))
    print(f"Processed: {len(results)} websites for {args.code}")
    
    if args.filter_dns == "filter_dns":
        with open(os.path.join("results", args.code, "repeated_output.json"), "w") as output_file:
            filtered_results, repeated = filter_unique_domains(results, False)
            
            json.dump(repeated, output_file)
            print(f"Filtered: {len(results) - len(filtered_results)} websites")
    
    json.dump(results, f)