import json
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import sys

from src.steps.analysis.helpers import process_experiment_cdn, find_experiment_files

# ===================
# Argument Parser
# ===================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot CDN usage by provider with optional accumulation and VPN filter.")
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
    file_types = ["cdn.json", "locedge.json", "provider.json"]
    experiment_paths = find_experiment_files(base_path, file_types, args.vpn)

    aggregated_counts = defaultdict(lambda: {"cdn": 0, "no_cdn": 0})
    total_unknown = 0

    for cdn_path, locedge_path, provider_path in experiment_paths:
        with open(cdn_path) as f:
            cdn_data = json.load(f)
        with open(locedge_path) as f:
            locedge_data = json.load(f)
        with open(provider_path) as f:
            provider_data = json.load(f)

        # Process this experiment to get provider counts
        provider_counts = defaultdict(lambda: {"cdn": 0, "no_cdn": 0})
        unknown_count = process_experiment_cdn(cdn_data, locedge_data, provider_data, provider_counts)
        total_unknown += unknown_count
        
        # Add to aggregated data
        for provider, counts in provider_counts.items():
            aggregated_counts[provider]["cdn"] += counts["cdn"]
            aggregated_counts[provider]["no_cdn"] += counts["no_cdn"]

    # Use helper function for aggregation, but keep the special Amazon logic
    final_counts = defaultdict(lambda: {"cdn": 0, "no_cdn": 0})
    others_count = {"cdn": 0, "no_cdn": 0}
    others = 0

    for provider, counts in aggregated_counts.items():
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

    sorted_providers = sorted(final_counts.keys(), key=lambda p: final_counts[p]["cdn"] + final_counts[p]["no_cdn"], reverse=True)

    if others_count["cdn"] + others_count["no_cdn"] > 0:
        others_label = f"others ({others})"
        final_counts[others_label] = others_count
        sorted_providers.append(others_label)

    total_counts = [final_counts[p]["cdn"] + final_counts[p]["no_cdn"] for p in sorted_providers]
    unknown_pct = (total_unknown / sum(total_counts)) * 100 if sum(total_counts) > 0 else 0

    if args.accumulated:
        accumulated_counts = [sum(total_counts[:i + 1]) for i in range(len(total_counts))]
        accumulated_pct = [val / accumulated_counts[-1] * 100 for val in accumulated_counts]

        accumulated_by_type = []
        for i, p in enumerate(sorted_providers):
            if i == 0:
                accumulated_by_type.append((final_counts[p]["cdn"], final_counts[p]["no_cdn"]))
            else:
                accumulated_by_type.append((
                    accumulated_by_type[i - 1][0] + final_counts[p]["cdn"],
                    accumulated_by_type[i - 1][1] + final_counts[p]["no_cdn"]
                ))

        cdn_acc_pct = [c / accumulated_counts[-1] * 100 for c, _ in accumulated_by_type]
        no_cdn_acc_pct = [n / accumulated_counts[-1] * 100 for _, n in accumulated_by_type]

        # Plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(sorted_providers, cdn_acc_pct, color="blue", label="CDN")
        plt.bar(sorted_providers, no_cdn_acc_pct, bottom=cdn_acc_pct, color="red", label="Self-hosted")

        for bar, pct in zip(bars, accumulated_pct):
            plt.text(bar.get_x() + bar.get_width() / 2, pct + 2, f"{pct:.1f}%", ha="center", fontsize=8)
    else:
        cdn_pct = [(final_counts[p]["cdn"] / t) * 100 if t else 0 for p, t in zip(sorted_providers, total_counts)]
        no_cdn_pct = [(final_counts[p]["no_cdn"] / t) * 100 if t else 0 for p, t in zip(sorted_providers, total_counts)]

        # Plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(sorted_providers, cdn_pct, color="blue", label="CDN")
        plt.bar(sorted_providers, no_cdn_pct, bottom=cdn_pct, color="red", label="Self-hosted")

        for bar, total in zip(bars, total_counts):
            plt.text(bar.get_x() + bar.get_width() / 2, 102, str(total), ha="center", fontsize=8, fontweight="bold")

    title_note = f" - VPN: {args.vpn}" if args.vpn else ""
    acc_note = " - Accumulated" if args.accumulated else ""
    unknown_note = f" | Unknown: {unknown_pct:.1f}%" if total_unknown > 0 else ""

    plt.title(f"CDN Usage by Provider for {args.country}{title_note}{acc_note}{unknown_note}")
    plt.xlabel("Provider")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 110)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    output_path = Path(f"results/{args.code}/results/locality")
    output_path.mkdir(parents=True, exist_ok=True)
    
    suffix = "accumulated" if args.accumulated else "stacked"
    vpn_suffix = f"_vpn_{args.vpn}" if args.vpn else ""
    out_file = output_path / f"providers_{suffix}{vpn_suffix}.png"

    if args.save:
        plt.savefig(out_file)
    else:
        plt.show()


if __name__ == "__main__":
    main()
