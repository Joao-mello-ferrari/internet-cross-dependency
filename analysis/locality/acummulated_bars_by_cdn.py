import json
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from glob import glob
import os

# Argument parser
parser = argparse.ArgumentParser(description="Plot accumulated CDN usage by provider.")
parser.add_argument("country", type=str, help="Country of the process")
parser.add_argument("save", type=str, help="Should save plot")
args = parser.parse_args()

# Initialize aggregated data
provider_counts = defaultdict(lambda: {"cdn": 0, "no_cdn": 0})
total_unknown = 0

# Find all experiments using glob
experiment_paths = set(os.path.dirname(path) for path in glob(f"results/{args.country}/locality/**/cdn.json", recursive=True))

for experiment_path in experiment_paths:
    try:
        with open(os.path.join(experiment_path, "cdn.json")) as f:
            cdn_data = json.load(f)
        with open(os.path.join(experiment_path, "locedge.json")) as f:
            locedge_data = json.load(f)
        with open(os.path.join(experiment_path, "provider.json")) as f:
            provider_data = json.load(f)
    except Exception as e:
        print(f"Skipping {experiment_path} due to error: {e}")
        continue

    for domain, cdn_string in cdn_data.items():
        by_whois_provider = provider_data.get(domain)
        cdn_providers = set(cdn_string.replace("'", "").split(", ")) if cdn_string else set()
        no_cdn_providers = set(locedge_data.get(domain, {}).get("provider", []))
        if by_whois_provider:
            no_cdn_providers.add(by_whois_provider)

        if not cdn_providers and not no_cdn_providers:
            total_unknown += 1
            continue

        if cdn_providers:
            for provider in map(str.lower, cdn_providers):
                if provider == "aws (not cdn)": continue
                provider_counts[provider]["cdn"] += 1
        else:
            for provider in map(str.lower, no_cdn_providers):
                if provider == "aws (not cdn)": continue
                provider_counts[provider]["no_cdn"] += 1

# Aggregate smaller providers into "others"
aggregated_counts = defaultdict(lambda: {"cdn": 0, "no_cdn": 0})
others_count = {"cdn": 0, "no_cdn": 0}
others = 0

for provider, counts in provider_counts.items():
    total = counts["cdn"] + counts["no_cdn"]
    if total < 5:
        others_count["cdn"] += counts["cdn"]
        others_count["no_cdn"] += counts["no_cdn"]
        others += 1
    elif provider in ["cloudfront", "amazon aws", "amazon"]:
        aggregated_counts["amazon"]["cdn"] += counts["cdn"]
        aggregated_counts["amazon"]["no_cdn"] += counts["no_cdn"]
    else:
        aggregated_counts[provider] = counts

# Sort by total
sorted_providers = sorted(
    aggregated_counts.keys(),
    key=lambda p: aggregated_counts[p]["cdn"] + aggregated_counts[p]["no_cdn"],
    reverse=True
)

if others_count["cdn"] + others_count["no_cdn"] > 0:
    aggregated_counts[f"others ({others})"] = others_count
    sorted_providers.append(f"others ({others})")


# Compute accumulated percentages
total_counts = [aggregated_counts[p]["cdn"] + aggregated_counts[p]["no_cdn"] for p in sorted_providers]
accumulated_counts = [sum(total_counts[:i + 1]) for i in range(len(total_counts))]
accumulated_percentages = [accumulated_counts[i] / sum(total_counts) * 100 for i in range(len(total_counts))]

accumulated_counts_by_cdn_usage = []
for idx, p in enumerate(sorted_providers):
    if idx == 0:
        accumulated_counts_by_cdn_usage.append((aggregated_counts[p]["cdn"], aggregated_counts[p]["no_cdn"]))
    else:
        accumulated_counts_by_cdn_usage.append((
            accumulated_counts_by_cdn_usage[idx - 1][0] + aggregated_counts[p]["cdn"],
            accumulated_counts_by_cdn_usage[idx - 1][1] + aggregated_counts[p]["no_cdn"]
        ))

accumulated_percentages_by_cdn_usage = [
    (cdn / accumulated_counts[-1] * 100, no_cdn / accumulated_counts[-1] * 100) 
    for cdn, no_cdn in accumulated_counts_by_cdn_usage
]
cdn_acumulated_percentages = [cdn for cdn, _ in accumulated_percentages_by_cdn_usage]
no_cdn_acumulated_percentages = [no_cdn for _, no_cdn in accumulated_percentages_by_cdn_usage]

print(accumulated_counts_by_cdn_usage)
# /print(total_counts[-1])

# Plot
plt.figure(figsize=(12, 6))
bars = plt.bar(sorted_providers, cdn_acumulated_percentages, color="blue", label="Accumulated Percentage")
bars_no_cdn = plt.bar(sorted_providers, no_cdn_acumulated_percentages, bottom=cdn_acumulated_percentages, color="red", label="No CDN")

for bar, percentage in zip(bars_no_cdn, accumulated_percentages):
    plt.text(bar.get_x() + bar.get_width() / 2, percentage + 2, f"{percentage:.1f}%", ha="center", fontsize=8)

plt.xlabel("Provider")
plt.ylabel("Accumulated Percentage (%)")
plt.title("Accumulated CDN/Non-CDN Usage by Provider")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 110)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()

if args.save.lower() == "true":
    plt.savefig(f"results/{args.country}/results/providers_accumulated.png")
else:
    plt.show()
