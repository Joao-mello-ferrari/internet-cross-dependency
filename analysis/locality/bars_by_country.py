import json
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import glob
import os

# Argument parser
parser = argparse.ArgumentParser(description="Plot a stacked bar chart of anycast usage.")
parser.add_argument("country", type=str, help="Country of the process")
parser.add_argument("save", type=str, help="Should save plot")
parser.add_argument("--aggregate", action="store_true", help="Aggregate results across experiments")
args = parser.parse_args()

# Recursively find all experiment folders with the required JSONs
experiment_paths = []
for location_path in glob.glob(f"results/{args.country}/locality/**/location.json", recursive=True):
    experiment_dir = os.path.dirname(location_path)
    #provider_path = os.path.join(experiment_dir, "country.json")
    locedge_path = os.path.join(experiment_dir, "locedge.json")
    if os.path.exists(locedge_path):
        experiment_paths.append((location_path, locedge_path))

# Master country count map
aggregated_counts = defaultdict(lambda: {"anycast": 0, "no_anycast": 0})
total_unknown = 0

def process_experiment(location_data, locedge_data, country_counts):
    local_unknown = 0
    for domain, location in location_data.items():
        content_locality_by_locedge = locedge_data.get(domain, {}).get("contentLocality", None)

        if location is None and content_locality_by_locedge is None:
            local_unknown += 1
            continue

        if content_locality_by_locedge is not None:
            country_counts[content_locality_by_locedge]["no_anycast"] += 1
            continue

        if "- anycast" in location:
            raw_location = location.replace(" - anycast", "")
            country_counts[raw_location]["anycast"] += 1
        else:
            country_counts[location]["no_anycast"] += 1

    return local_unknown

# Process each experiment
for location_path, locedge_path in experiment_paths:
    with open(location_path) as f:
        location_data = json.load(f)
    with open(locedge_path) as f:
        locedge_data = json.load(f)

    country_counts = aggregated_counts if args.aggregate else defaultdict(lambda: {"anycast": 0, "no_anycast": 0})
    total_unknown += process_experiment(location_data, locedge_data, country_counts)

# Use either aggregated or last country_counts
country_counts = aggregated_counts if args.aggregate else country_counts

# Combine similar and small countries
final_counts = defaultdict(lambda: {"anycast": 0, "no_anycast": 0})
others_count = {"anycast": 0, "no_anycast": 0}
others = 0

for country, counts in country_counts.items():
    total = counts["anycast"] + counts["no_anycast"]
    name = country.lower()
    if total < 5:
        others_count["anycast"] += counts["anycast"]
        others_count["no_anycast"] += counts["no_anycast"]
        others += 1
    else:
        final_counts[country] = counts

# Sorting
sorted_countries = sorted(final_counts.keys(), key=lambda p: final_counts[p]["anycast"] + final_counts[p]["no_anycast"], reverse=True)

# Add "Others"
if others_count["anycast"] + others_count["no_anycast"] > 0:
    final_counts[f"others ({others})"] = others_count
    sorted_countries.append(f"others ({others})")

# Add unknown
sorted_countries.append("unknown")

# Percentages and totals
total_counts = [final_counts[p]["anycast"] + final_counts[p]["no_anycast"] for p in sorted_countries[:-1]]
anycast_percentages = [(final_counts[p]["anycast"] / total) * 100 if total else 0 for p, total in zip(sorted_countries[:-1], total_counts)]
no_anycast_percentages = [(final_counts[p]["no_anycast"] / total) * 100 if total else 0 for p, total in zip(sorted_countries[:-1], total_counts)]

# Add unknown
anycast_percentages.append(0)
no_anycast_percentages.append(0)
unknown_percentage = 100 if total_unknown > 0 else 0
total_counts.append(total_unknown)

# Plot
plt.figure(figsize=(12, 6))
bars_no_anycast = plt.bar(sorted_countries, no_anycast_percentages, color="blue", label="No anycast")
bars_anycast = plt.bar(sorted_countries, anycast_percentages, bottom=no_anycast_percentages, color="red", label="Anycast")
bars_unknown = plt.bar(sorted_countries, [0]*(len(sorted_countries)-1)+[unknown_percentage], 
                       bottom=anycast_percentages, color="black", label="Self Hosted | Unknown")

# Counts
for bar, total in zip(bars_anycast, total_counts):
    plt.text(bar.get_x() + bar.get_width()/2, 102, str(total), ha="center", fontsize=8, fontweight="bold")

plt.xlabel("Origin Country")
plt.ylabel("Percentage (%)")
plt.title(f"Anycast vs Non-anycast Usage by Country ({'Aggregated' if args.aggregate else 'Single Experiment'}) - {args.country}")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 110)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

if args.save == "true":
    plt.savefig(f"results/{args.country}/results/aggregated_countries.png" if args.aggregate else f"results/{args.country}/results/last_experiment_countries.png")
else:
    plt.show()
