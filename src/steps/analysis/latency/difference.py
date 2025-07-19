# --- Imports ---
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter
from src.steps.analysis.latency.data_parse_helpers import setup_paths, get_rtts

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Generate RTT plots comparing IPv4 and IPv6 point by point.")
parser.add_argument("--country", type=str, required=True, help="Country label")
parser.add_argument("--code", type=str, required=True, help="Country code used for folder path")
parser.add_argument("--save", action='store_true', help="Save the generated figures")
args = parser.parse_args()


# --- Plotting Function ---
def plot(ipv4_rtts, ipv6_rtts, label, subdir):
    out_path = base_path / "results" / subdir
    out_path.mkdir(parents=True, exist_ok=True)

    rtt_diff = ipv6_rtts - ipv4_rtts
    data = pd.DataFrame({'ipv4': ipv4_rtts, 'ipv6': ipv6_rtts, 'diff': rtt_diff})
    data_sorted = data.sort_values("diff").reset_index(drop=True)

    mean_diff = np.mean(rtt_diff)
    median_diff = np.median(rtt_diff)
    std_diff = np.std(rtt_diff)
    pct_v6_slower = np.mean(rtt_diff > 0) * 100
    pct_v4_slower = np.mean(rtt_diff < 0) * 100
    t_stat, p_val = stats.ttest_1samp(rtt_diff, 0)

    stats_text = (
        f"Mean: {mean_diff:.2f} ms\n"
        f"Median: {median_diff:.2f} ms\n"
        f"IPv6 slower: {pct_v6_slower:.1f}%\n"
        f"IPv4 slower: {pct_v4_slower:.1f}%\n"
        f"Significant: {'Yes (p<0.05)' if p_val < 0.05 else 'No (p≥0.05)'}\n"
        f"IPv6 Route Failures: {fail_count}"
    )

    # Histogram of differences
    plt.figure(figsize=(12, 6))
    bins = np.linspace(np.min(rtt_diff), np.max(rtt_diff), 30)
    plt.hist(rtt_diff, bins=bins, color="#3498db", alpha=0.7)
    plt.axvline(x=0, color="black", linestyle="--")
    plt.axvline(x=mean_diff, color="purple", linestyle="-", label=f"Mean: {mean_diff:.2f}")
    plt.axvline(x=median_diff, color="green", linestyle="-", label=f"Median: {median_diff:.2f}")
    plt.title(f"RTT Differences Histogram ({args.country}, {label})")
    plt.xlabel("IPv6 - IPv4 RTT (ms)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    if args.save:
        plt.savefig(out_path / "rtts_difference_histogram.png", dpi=300)
    else:
        plt.show()
        plt.close()

    # Sorted difference
    plt.figure(figsize=(14, 8))
    x = np.arange(len(data_sorted))
    plt.scatter(x, data_sorted["diff"], alpha=0.5, label="Sorted Diff", color="#3498db")
    if len(data_sorted) > 4:
        smoothed = savgol_filter(data_sorted["diff"], min(len(data_sorted) - 1 if len(data_sorted) % 2 == 0 else len(data_sorted), 15), 3)
        plt.plot(x, smoothed, color="#e74c3c", label="Smoothed Trend")
    plt.axhline(0, linestyle="--", color="black")
    plt.fill_between(x, mean_diff - std_diff, mean_diff + std_diff, color="purple", alpha=0.2, label="±1 Std Dev")
    plt.title(f"Sorted RTT Differences ({args.country}, {label})")
    plt.xlabel("Measurements (sorted)")
    plt.ylabel("IPv6 - IPv4 RTT (ms)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    if args.save:
        plt.savefig(out_path / "sorted_rtts_difference.png", dpi=300)
    else:
        plt.show()
        plt.close()

    # Absolute RTTs
    plt.figure(figsize=(14, 8))
    plt.plot(x, data_sorted["ipv4"], label="IPv4", marker='o', linestyle='-', alpha=0.7)
    plt.plot(x, data_sorted["ipv6"], label="IPv6", marker='o', linestyle='-', alpha=0.7)
    plt.fill_between(x, data_sorted["ipv4"], data_sorted["ipv6"],
                     where=(data_sorted["ipv6"] > data_sorted["ipv4"]),
                     color="#e74c3c", alpha=0.3, label="IPv6 Slower")
    plt.fill_between(x, data_sorted["ipv4"], data_sorted["ipv6"],
                     where=(data_sorted["ipv4"] >= data_sorted["ipv6"]),
                     color="#3498db", alpha=0.3, label="IPv4 Slower")
    cross_idx = data_sorted["diff"].ge(0).idxmax()
    plt.axvline(x=cross_idx, color="black", linestyle="--", label=f"Crossover @ {cross_idx}")
    plt.title(f"Absolute RTT Values by Difference ({args.country}, {label})")
    plt.xlabel("Measurements (sorted)")
    plt.ylabel("RTT (ms)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.text(0.02, 0.98,
             f"IPv4 better | IPv6 better\nLeft: {cross_idx} ({cross_idx/len(data_sorted)*100:.1f}%)\n"
             f"Right: {len(data_sorted) - cross_idx} ({(1 - cross_idx/len(data_sorted))*100:.1f}%)",
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()

    if args.save:
        plt.savefig(out_path / "rtts_by_difference.png", dpi=300)
    else:
        plt.show()
        plt.close()

# --- Load Latency Data ---
base_path, latency_ipv4_input, latency_ipv6_input, fail_ipv6_input = setup_paths(args.code)
ipv4_by_domain, ipv6_by_domain, ipv4_by_probe, ipv6_by_probe, fail_ipv6_route, domains_count = get_rtts(
    latency_ipv4_input,
    latency_ipv6_input,
    fail_ipv6_input,
)
fail_count = len(fail_ipv6_route)

# --- Generate Plots ---
plot(ipv4_by_domain, ipv6_by_domain, "aggregated by domain", "latency/by_domain")
plot(ipv4_by_probe, ipv6_by_probe, "aggregated by probe", "latency/by_probe")