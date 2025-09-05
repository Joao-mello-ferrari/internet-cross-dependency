# --- Imports ---
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import PercentFormatter
from src.steps.analysis.latency.data_parse_helpers import setup_paths, get_rtts

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Generate RTT CDF plots comparing IPv4 and IPv6 for a given country.")
parser.add_argument("--country", type=str, required=True, help="Country label")
parser.add_argument("--code", type=str, required=True, help="Country code used for folder path")
parser.add_argument("--save", action='store_true', help="Save the figure")
args = parser.parse_args()


# --- Plotting Function ---
def plot(ipv4_rtts, ipv6_rtts, fail_ipv6_route_annotation, subfolder, aggregation_type_label):
    out_path = base_path / "results" / subfolder / "cdf.png"
    plt.style.use("ggplot")
    sns.set(style="whitegrid", font_scale=1.2)
    colors = ["#3498db", "#e74c3c"]

    if len(ipv6_rtts) == 0:
        x_range = np.linspace(np.min(ipv4_rtts), np.max(ipv4_rtts), 1000)
    else:
        x_range = np.linspace(min(np.min(ipv4_rtts), np.min(ipv6_rtts)),
                              max(np.max(ipv4_rtts), np.max(ipv6_rtts)), 1000)

    #kde_v4 = stats.gaussian_kde(ipv4_rtts)
    #kde_v6 = stats.gaussian_kde(ipv6_rtts)
    #pdf_v4 = kde_v4(x_range)
    #pdf_v6 = kde_v6(x_range)
    #pdf_v4 /= np.sum(pdf_v4)
    #pdf_v6 /= np.sum(pdf_v6)

    if len(ipv6_rtts) != 0:
        _, t_pval = stats.ttest_ind(ipv4_rtts, ipv6_rtts, equal_var=False)

    # CDF
    ipv4_sorted = np.sort(ipv4_rtts)
    cdf_v4 = np.arange(1, len(ipv4_sorted) + 1) / len(ipv4_sorted)
    if len(ipv6_rtts) != 0:
        ipv6_sorted = np.sort(ipv6_rtts)
        cdf_v6 = np.arange(1, len(ipv6_sorted) + 1) / len(ipv6_sorted)

    plt.figure(figsize=(12, 8))
    plt.plot(ipv4_sorted, cdf_v4, label=f"IPv4 (mean={np.mean(ipv4_rtts):.2f}ms, samples={len(ipv4_rtts)})", color=colors[0], linewidth=2.5)
    if len(ipv6_rtts) != 0:
        plt.plot(ipv6_sorted, cdf_v6, label=f"IPv6 (mean={np.mean(ipv6_rtts):.2f}ms, samples={len(ipv6_rtts)})", color=colors[1], linewidth=2.5)

    plt.axhline(0.5, linestyle="--", color="gray", alpha=0.7)
    plt.axvline(np.median(ipv4_rtts), linestyle="--", color=colors[0], alpha=0.7)
    if len(ipv6_rtts) != 0:
        plt.axvline(np.median(ipv6_rtts), linestyle="--", color=colors[1], alpha=0.7)

    plt.text(np.median(ipv4_rtts) + 2, 0.52, f"IPv4 Median:\n{np.median(ipv4_rtts):.1f} ms", color=colors[0], fontsize=10)
    if len(ipv6_rtts) != 0:
        plt.text(np.median(ipv6_rtts) + 2, 0.48, f"IPv6 Median:\n{np.median(ipv6_rtts):.1f} ms", color=colors[1], fontsize=10)

    if len(ipv6_rtts) != 0:
        plt.title("CDF - IPv4 vs IPv6 | {} - Aggregated by {}".format(args.country, aggregation_type_label))
    else:
        plt.title("CDF - IPv4 | {} - Aggregated by {}".format(args.country, aggregation_type_label))
    plt.xlabel("Round Trip Time (ms)")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

    # IPv6 failure count and statistical diff annotations
    stat_text = f"IPv6 Routing Failures: {fail_ipv6_route_annotation}"
    if len(ipv6_rtts) != 0:
        stat_text += \
            f"Statistical Difference: {'Significant (p<0.05)' if t_pval < 0.05 else 'Not Significant (p≥0.05)'}\n"
    plt.text(0.02, 0.02, stat_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()

    if args.save:
        plt.savefig(out_path)
    else:
        plt.show()
        plt.close()

# --- Setup Paths and Load Data ---
base_path, latency_ipv4_input, latency_ipv6_input, fail_ipv6_input = setup_paths(args.code)
ipv4_by_domain, ipv6_by_domain, ipv4_by_probe, ipv6_by_probe, fail_ipv6_route, domains_count = get_rtts(
    latency_ipv4_input,
    latency_ipv6_input,
    fail_ipv6_input,
    False
)
fail_ipv6_route_annotation = "{} ({:.2f}% of domains)".format(len(fail_ipv6_route), (len(fail_ipv6_route) / domains_count) * 100)

# --- Plot Results ---
#plot(ipv4_by_domain, ipv6_by_domain, fail_ipv6_route_annotation, "latency/by_domain", "domain")
plot(ipv4_by_probe, ipv6_by_probe, fail_ipv6_route_annotation, "latency/by_probe", "probe")
