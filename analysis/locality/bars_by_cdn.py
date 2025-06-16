import json
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import glob
import os

# Argument parser
parser = argparse.ArgumentParser(description="Plot a stacked bar chart of CDN usage.")
parser.add_argument("country", type=str, help="Country of the process")
parser.add_argument("save", type=str, help="Should save plot")
parser.add_argument("--aggregate", action="store_true", help="Aggregate results across experiments")
args = parser.parse_args()

# Recursively find all experiment folders with the required JSONs
experiment_paths = []
for cdn_path in glob.glob(f"results/{args.country}/locality/**/cdn.json", recursive=True):
    experiment_dir = os.path.dirname(cdn_path)
    provider_path = os.path.join(experiment_dir, "provider.json")
    locedge_path = os.path.join(experiment_dir, "locedge.json")
    if os.path.exists(provider_path) and os.path.exists(locedge_path):
        experiment_paths.append((cdn_path, locedge_path, provider_path))

# Master provider count map
aggregated_counts = defaultdict(lambda: {"cdn": 0, "no_cdn": 0})
total_unknown = 0

def process_experiment(cdn_data, locedge_data, provider_data, provider_counts):
    local_unknown = 0
    for domain, cdn_string in cdn_data.items():
        by_whois_provider = provider_data.get(domain)
        cdn_providers = set(cdn_string.replace("'", "").split(", ")) if cdn_string else []
        no_cdn_providers = set(locedge_data.get(domain, {}).get("provider", []) + [by_whois_provider] if by_whois_provider is not None else [])

        if len(cdn_providers) == 0 and len(no_cdn_providers) == 0:
            local_unknown += 1
            continue

        if len(cdn_providers) > 0:
            cdn_providers = set(map(str.lower, cdn_providers))
            for provider in cdn_providers:
                if provider == "aws (not cdn)": continue
                provider_counts[provider]["cdn"] += 1
        else:
            for provider in list(map(str.lower, no_cdn_providers)):
                if provider == "aws (not cdn)": continue
                provider_counts[provider]["no_cdn"] += 1
    return local_unknown

# Process each experiment
for cdn_path, locedge_path, provider_path in experiment_paths:
    with open(cdn_path) as f:
        cdn_data = json.load(f)
    with open(locedge_path) as f:
        locedge_data = json.load(f)
    with open(provider_path) as f:
        provider_data = json.load(f)

    provider_counts = aggregated_counts if args.aggregate else defaultdict(lambda: {"cdn": 0, "no_cdn": 0})
    total_unknown += process_experiment(cdn_data, locedge_data, provider_data, provider_counts)

# Use either aggregated or last provider_counts
provider_counts = aggregated_counts if args.aggregate else provider_counts

# Combine similar and small providers
final_counts = defaultdict(lambda: {"cdn": 0, "no_cdn": 0})
others_count = {"cdn": 0, "no_cdn": 0}
others = 0

for provider, counts in provider_counts.items():
    total = counts["cdn"] + counts["no_cdn"]
    name = provider.lower()
    if total < 5:
        others_count["cdn"] += counts["cdn"]
        others_count["no_cdn"] += counts["no_cdn"]
        others += 1
    elif name in ["cloudfront", "amazon aws", "amazon"]:
        final_counts["amazon"]["cdn"] += counts["cdn"]
        final_counts["amazon"]["no_cdn"] += counts["no_cdn"]
    else:
        final_counts[provider] = counts

# Sorting
sorted_providers = sorted(final_counts.keys(), key=lambda p: final_counts[p]["cdn"] + final_counts[p]["no_cdn"], reverse=True)

# Add "Others"
if others_count["cdn"] + others_count["no_cdn"] > 0:
    final_counts[f"others ({others})"] = others_count
    sorted_providers.append(f"others ({others})")

# Add unknown
sorted_providers.append("unknown")

# Percentages and totals
total_counts = [final_counts[p]["cdn"] + final_counts[p]["no_cdn"] for p in sorted_providers[:-1]]
cdn_percentages = [(final_counts[p]["cdn"] / total) * 100 if total else 0 for p, total in zip(sorted_providers[:-1], total_counts)]
no_cdn_percentages = [(final_counts[p]["no_cdn"] / total) * 100 if total else 0 for p, total in zip(sorted_providers[:-1], total_counts)]

# Add unknown
cdn_percentages.append(0)
no_cdn_percentages.append(0)
unknown_percentage = 100 if total_unknown > 0 else 0
total_counts.append(total_unknown)

# Plot
plt.figure(figsize=(12, 6))
bars_no_cdn = plt.bar(sorted_providers, no_cdn_percentages, bottom=cdn_percentages, color="red", label="No CDN")
bars_cdn = plt.bar(sorted_providers, cdn_percentages, color="blue", label="CDN")
bars_unknown = plt.bar(sorted_providers, [0]*(len(sorted_providers)-1)+[unknown_percentage], 
                       bottom=cdn_percentages, color="black", label="Self Hosted | Unknown")

# Counts
for bar, total in zip(bars_cdn, total_counts):
    plt.text(bar.get_x() + bar.get_width()/2, 102, str(total), ha="center", fontsize=8, fontweight="bold")

plt.xlabel("Provider")
plt.ylabel("Percentage (%)")
plt.title(f"CDN vs Non-CDN Usage by Provider ({'Aggregated' if args.aggregate else 'Single Experiment'})")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 110)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

if args.save == "true":
    plt.savefig(f"results/{args.country}/results/aggregated_providers.png" if args.aggregate else f"results/{args.country}/results/last_experiment_providers.png")
else:
    plt.show()
