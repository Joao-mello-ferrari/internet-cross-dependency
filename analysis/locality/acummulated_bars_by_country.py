import json
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from glob import glob
import os

# Argument parser
parser = argparse.ArgumentParser(description="Plot accumulated anycast usage by country.")
parser.add_argument("country", type=str, help="Country of the process")
parser.add_argument("save", type=str, help="Should save plot")
args = parser.parse_args()

# Initialize aggregated data
country_counts = defaultdict(lambda: {"anycast": 0, "no_anycast": 0})
total_unknown = 0

# Find all experiments using glob
experiment_paths = set(os.path.dirname(path) for path in glob(f"results/{args.country}/locality/**/location.json", recursive=True))

for experiment_path in experiment_paths:
    try:
        with open(os.path.join(experiment_path, "locedge.json")) as f:
            locedge_data = json.load(f)
        with open(os.path.join(experiment_path, "location.json")) as f:
            location_data = json.load(f)
    except Exception as e:
        print(f"Skipping {experiment_path} due to error: {e}")
        continue

    for domain, location in location_data.items():
        content_locality_by_locedge = locedge_data.get(domain, {}).get("contentLocality", None)

        if location is None and content_locality_by_locedge is None:
            local_unknown += 1
            continue

        if content_locality_by_locedge is not None:
            country_counts[content_locality_by_locedge]["no_anycast"] += 1
            continue

        if "- anycast" in location:
            raw_location = location.replace("- anycast", "")
            country_counts[raw_location]["anycast"] += 1
        else:
            country_counts[location]["no_anycast"] += 1

# Aggregate smaller countries into "others"
aggregated_counts = defaultdict(lambda: {"anycast": 0, "no_anycast": 0})
others_count = {"anycast": 0, "no_anycast": 0}
others = 0

for country, counts in country_counts.items():
    total = counts["anycast"] + counts["no_anycast"]
    if total < 5:
        others_count["anycast"] += counts["anycast"]
        others_count["no_anycast"] += counts["no_anycast"]
        others += 1
    else:
        aggregated_counts[country] = counts

print(country_counts)
print(aggregated_counts)

# Sort by total
sorted_countries = sorted(
    aggregated_counts.keys(),
    key=lambda p: aggregated_counts[p]["anycast"] + aggregated_counts[p]["no_anycast"],
    reverse=True
)

if others_count["anycast"] + others_count["no_anycast"] > 0:
    aggregated_counts[f"others ({others})"] = others_count
    sorted_countries.append(f"others ({others})")


# Compute accumulated percentages
total_counts = [aggregated_counts[p]["anycast"] + aggregated_counts[p]["no_anycast"] for p in sorted_countries]
accumulated_counts = [sum(total_counts[:i + 1]) for i in range(len(total_counts))]
accumulated_percentages = [accumulated_counts[i] / sum(total_counts) * 100 for i in range(len(total_counts))]

accumulated_counts_by_anycast_usage = []
for idx, p in enumerate(sorted_countries):
    if idx == 0:
        accumulated_counts_by_anycast_usage.append((aggregated_counts[p]["anycast"], aggregated_counts[p]["no_anycast"]))
    else:
        accumulated_counts_by_anycast_usage.append((
            accumulated_counts_by_anycast_usage[idx - 1][0] + aggregated_counts[p]["anycast"],
            accumulated_counts_by_anycast_usage[idx - 1][1] + aggregated_counts[p]["no_anycast"]
        ))

accumulated_percentages_by_anycast_usage = [
    (anycast / accumulated_counts[-1] * 100, no_anycast / accumulated_counts[-1] * 100) 
    for anycast, no_anycast in accumulated_counts_by_anycast_usage
]
anycast_acumulated_percentages = [anycast for anycast, _ in accumulated_percentages_by_anycast_usage]
no_anycast_acumulated_percentages = [no_anycast for _, no_anycast in accumulated_percentages_by_anycast_usage]

print(accumulated_counts_by_anycast_usage)
# /print(total_counts[-1])

# Plot
plt.figure(figsize=(12, 6))
bars_no_anycast = plt.bar(sorted_countries, no_anycast_acumulated_percentages, color="blue", label="No anycast")
bars = plt.bar(sorted_countries, anycast_acumulated_percentages, bottom=no_anycast_acumulated_percentages, color="red", label="Accumulated Percentage")

for bar, percentage in zip(bars_no_anycast, accumulated_percentages):
    plt.text(bar.get_x() + bar.get_width() / 2, percentage + 2, f"{percentage:.1f}%", ha="center", fontsize=8)

plt.xlabel("country")
plt.ylabel("Accumulated Percentage (%)")
plt.title("Accumulated anycast/Non-anycast Usage by country - " + args.country)
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 110)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()

if args.save.lower() == "true":
    plt.savefig(f"results/{args.country}/results/countries_accumulated.png")
else:
    plt.show()
