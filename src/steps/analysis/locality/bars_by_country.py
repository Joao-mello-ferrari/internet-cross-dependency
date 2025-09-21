import json
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

from src.steps.analysis.helpers import process_experiment_country, find_experiment_files, aggregate_small_items

# ===================
# Argument Parser
# ===================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot anycast usage by country with optional accumulation and VPN filter.")
    parser.add_argument("--country", type=str, required=True, help="Country label")
    parser.add_argument("--code", type=str, required=True, help="Country code used for folder path")
    parser.add_argument("--vpn", type=str, default=None, help="Optional VPN country code to filter")
    parser.add_argument("--save", action="store_true", help="Save the figure: 'true' or 'false'")
    parser.add_argument("--accumulated", action="store_true", help="Generate an accumulated bar plot")
    return parser.parse_args()


# ===================
# Main Function
# ===================
def main():
    args = parse_arguments()
    base_path = Path(f"results/{args.code}/locality")
    # Find all experiment files for this country
    file_types = ["location.json", "locedge.json"]
    experiment_paths = find_experiment_files(base_path, file_types, args.vpn)

    aggregated_counts = defaultdict(lambda: {"anycast": 0, "no_anycast": 0})
    total_unknown = 0

    for location_path, locedge_path in experiment_paths:
        with open(location_path) as f:
            location_data = json.load(f)
        with open(locedge_path) as f:
            locedge_data = json.load(f)

        # Process this experiment to get country counts
        country_counts = defaultdict(lambda: {"anycast": 0, "no_anycast": 0})
        unknown_count = process_experiment_country(location_data, locedge_data, country_counts, consider_anycast=True)
        total_unknown += unknown_count
        
        # Add to aggregated data
        for country, counts in country_counts.items():
            aggregated_counts[country]["anycast"] += counts["anycast"]
            aggregated_counts[country]["no_anycast"] += counts["no_anycast"]

    # Use helper function for aggregation
    final_counts, _ = aggregate_small_items(aggregated_counts, threshold=5, others_label="others")

    sorted_countries = sorted(
        final_counts.keys(),
        key=lambda p: final_counts[p]["anycast"] + final_counts[p]["no_anycast"],
        reverse=True
    )

    total_counts = [final_counts[c]["anycast"] + final_counts[c]["no_anycast"] for c in sorted_countries]
    unknown_pct = (total_unknown / sum(total_counts)) * 100 if sum(total_counts) > 0 else 0

    if args.accumulated:
        accumulated_counts = [sum(total_counts[:i + 1]) for i in range(len(total_counts))]
        accumulated_pct = [val / accumulated_counts[-1] * 100 for val in accumulated_counts]

        accumulated_by_type = []
        for i, c in enumerate(sorted_countries):
            if i == 0:
                accumulated_by_type.append((final_counts[c]["anycast"], final_counts[c]["no_anycast"]))
            else:
                accumulated_by_type.append((
                    accumulated_by_type[i - 1][0] + final_counts[c]["anycast"],
                    accumulated_by_type[i - 1][1] + final_counts[c]["no_anycast"]
                ))

        anycast_acc_pct = [a / accumulated_counts[-1] * 100 for a, _ in accumulated_by_type]
        no_anycast_acc_pct = [n / accumulated_counts[-1] * 100 for _, n in accumulated_by_type]

        # Plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(sorted_countries, no_anycast_acc_pct, color="blue", label="Unicast")
        plt.bar(sorted_countries, anycast_acc_pct, bottom=no_anycast_acc_pct, color="red", label="Anycast")

        for bar, pct in zip(bars, accumulated_pct):
            plt.text(bar.get_x() + bar.get_width() / 2, pct + 2, f"{pct:.1f}%", ha="center", fontsize=8)
    else:
        anycast_pct = [(final_counts[c]["anycast"] / t) * 100 if t else 0 for c, t in zip(sorted_countries, total_counts)]
        no_anycast_pct = [(final_counts[c]["no_anycast"] / t) * 100 if t else 0 for c, t in zip(sorted_countries, total_counts)]

        # Plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(sorted_countries, no_anycast_pct, color="blue", label="Unicast")
        plt.bar(sorted_countries, anycast_pct, bottom=no_anycast_pct, color="red", label="Anycast")

        for bar, total in zip(bars, total_counts):
            plt.text(bar.get_x() + bar.get_width() / 2, 102, str(total), ha="center", fontsize=8, fontweight="bold")

    title_note = f" - VPN: {args.vpn}" if args.vpn else ""
    acc_note = " - Accumulated" if args.accumulated else ""
    unknown_note = f" | Unknown: {unknown_pct:.1f}%" if total_unknown > 0 else ""

    plt.title(f"Country Dependency for {args.country}{title_note}{acc_note}{unknown_note}")
    plt.xlabel("Origin Country")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 110)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    output_path = Path(f"results/{args.code}/results/locality")
    output_path.mkdir(parents=True, exist_ok=True)
    suffix = "accumulated" if args.accumulated else "stacked"
    vpn_suffix = f"_vpn_{args.vpn}" if args.vpn else ""
    out_file = output_path / f"countries_{suffix}{vpn_suffix}.png"

    if args.save:
        plt.savefig(out_file)
    else:
        plt.show()


if __name__ == "__main__":
    main()
