import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from statistics import median
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import jensenshannon
from matplotlib.gridspec import GridSpec
import argparse
import json

parser = argparse.ArgumentParser(description="Process CDN origin for websites.")
parser.add_argument("country", type=str, help="Country name to read and write JSON files")
parser.add_argument('agg_func', type=str, help='Base longitude from VPN')
args = parser.parse_args()

base_path = Path(f"/Users/joaomello/Desktop/tcc/results/{args.country}")
(base_path / "results" / "by_domain").mkdir(parents=True, exist_ok=True)
(base_path / "results" / "by_probe_domain").mkdir(parents=True, exist_ok=True)

latency_ipv4_input = base_path / "latency" / f"latency_ipv4.json"
latency_ipv6_input = base_path / "latency" / f"latency_ipv6.json"
fail_ipv6_input = base_path / "latency" / f"fail_ipv6_route.json"

ipv4_rtts = []
ipv6_rtts = []
fail_ipv6_route = []

# Save results
def _filter_by_probe(ipv4_domain_probes, ipv6_domain_probes):
    ipv4_results, ipv6_results, ipv4_error, ipv6_error = [], [], 0, 0
    for probe_ipv4, probe_ipv6 in zip(ipv4_domain_probes, ipv6_domain_probes):
        filtered_ipv4 = list(filter(lambda result: result != None and result != 0, probe_ipv4))
        filtered_ipv6 = list(filter(lambda result: result != None and result != 0, probe_ipv6))

        if len(filtered_ipv4) == 0:
            ipv4_error += 1
        if len(filtered_ipv6) == 0:
            ipv6_error += 1
        
        if len(filtered_ipv4) > 0:
            ipv4_results.append(median(filtered_ipv4))

        if len(filtered_ipv6) > 0:  
            ipv6_results.append(median(filtered_ipv6))

    return ipv4_results, ipv6_results, ipv4_error, ipv6_error

def _filter(ipv4_domain_probes, ipv6_domain_probes):
    ipv4_results, ipv6_results = [], []
    for probe_ipv4, probe_ipv6 in zip(ipv4_domain_probes, ipv6_domain_probes):
        filtered_ipv4 = list(filter(lambda result: result != None and result != 0, probe_ipv4))
        filtered_ipv6 = list(filter(lambda result: result != None and result != 0, probe_ipv6))

        if len(filtered_ipv4) > 0: 
            ipv4_results.append(median(filtered_ipv4))
        if len(filtered_ipv6) > 0:  
            ipv6_results.append(median(filtered_ipv6))
    
    domain_ipv4_median = [median(ipv4_results)] if len(ipv4_results) > 0 else []
    domain_ipv6_median = [median(ipv6_results)] if len(ipv6_results) > 0 else []

    return domain_ipv4_median, domain_ipv6_median


ipv4_rtts_by_probe_domain = []
ipv6_rtts_by_probe_domain = []
ipv4_rtts_by_domain = []
ipv6_rtts_by_domain = []

with open(latency_ipv4_input, "r") as f, open(latency_ipv6_input, "r") as f1:
    ipv4_rtts_raw = json.load(f)
    ipv6_rtts_raw = json.load(f1)
    
    domains = list(set(ipv4_rtts_raw.keys()).union(set(ipv6_rtts_raw.keys())))
    for domain in domains:
        if domain not in ipv4_rtts_raw or domain not in ipv6_rtts_raw:
            continue

        ipv4_rtts_by_domain_result, ipv6_rtts_by_domain_result = _filter(ipv4_rtts_raw.get(domain), ipv6_rtts_raw.get(domain))
        ipv4_domain_probes_result, ipv6_domain_probes_result, ipv4_error, ipv6_error = _filter_by_probe(ipv4_rtts_raw.get(domain), ipv6_rtts_raw.get(domain))
        
        ipv4_rtts_by_domain += ipv4_rtts_by_domain_result
        ipv6_rtts_by_domain += ipv6_rtts_by_domain_result
        ipv4_rtts_by_probe_domain += ipv4_domain_probes_result
        ipv6_rtts_by_probe_domain += ipv6_domain_probes_result
    

    # Convert lists to numpy arrays
    ipv4_rtts_by_domain = np.array(ipv4_rtts_by_domain)
    ipv6_rtts_by_domain = np.array(ipv6_rtts_by_domain)
    ipv4_rtts_by_probe_domain = np.array(ipv4_rtts_by_probe_domain)
    ipv6_rtts_by_probe_domain = np.array(ipv6_rtts_by_probe_domain)

    print(ipv4_rtts_by_domain)
    print(ipv6_rtts_by_domain)
    print(ipv4_rtts_by_probe_domain)
    print(ipv6_rtts_by_probe_domain)

with open(fail_ipv6_input, "r") as f:
    fail_ipv6_route = json.load(f)



# PLOT GENERATION
def plot(ipv4_rtts, ipv6_rtts, subpath):
    rtt_cdf = base_path / "results" / subpath / f"cdf.png"
    
    # Set the visual styles
    plt.style.use('ggplot')
    sns.set(style="whitegrid", font_scale=1.2)
    colors = ["#3498db", "#e74c3c"]  # Blue for IPv4, Red for IPv6

    # Calculate common range for both distributions
    x_min = min(np.min(ipv4_rtts), np.min(ipv6_rtts))
    x_max = max(np.max(ipv4_rtts), np.max(ipv6_rtts))
    x_range = np.linspace(x_min, x_max, 1000)

    # Generate kernel density estimations for smooth curves
    ipv4_kde = stats.gaussian_kde(ipv4_rtts)
    ipv6_kde = stats.gaussian_kde(ipv6_rtts)

    ipv4_density = ipv4_kde(x_range)
    ipv6_density = ipv6_kde(x_range)

    # Calculate statistical distances
    # Normalize to ensure densities sum to 1
    ipv4_density_norm = ipv4_density / np.sum(ipv4_density)
    ipv6_density_norm = ipv6_density / np.sum(ipv6_density)

    # Calculate Jensen-Shannon Distance (square root of divergence)
    js_distance = jensenshannon(ipv4_density_norm, ipv6_density_norm)
    # Calculate Wasserstein Distance (Earth Mover's Distance)
    w_distance = stats.wasserstein_distance(ipv4_rtts, ipv6_rtts)
    # Calculate Kolmogorov-Smirnov distance and p-value
    ks_stat, ks_pvalue = stats.ks_2samp(ipv4_rtts, ipv6_rtts)


    # Calculate basic statistics
    ipv4_mean = np.mean(ipv4_rtts)
    ipv6_mean = np.mean(ipv6_rtts)
    ipv4_median = np.median(ipv4_rtts)
    ipv6_median = np.median(ipv6_rtts)
    ipv4_std = np.std(ipv4_rtts)
    ipv6_std = np.std(ipv6_rtts)

    # Calculate statistical differences
    t_stat, p_value = stats.ttest_ind(ipv4_rtts, ipv6_rtts, equal_var=False)


    # Create a CDF plot with distance visualization
    plt.figure(figsize=(12, 8))

    # Sort the data for CDF calculation
    ipv4_sorted = np.sort(ipv4_rtts)
    ipv6_sorted = np.sort(ipv6_rtts)

    # Calculate the cumulative probabilities
    ipv4_y = np.arange(1, len(ipv4_sorted) + 1) / len(ipv4_sorted)
    ipv6_y = np.arange(1, len(ipv6_sorted) + 1) / len(ipv6_sorted)

    # Plot the CDFs
    plt.plot(ipv4_sorted, ipv4_y, label=f'IPv4 (mean={ipv4_mean:.2f}ms)', linewidth=2.5, color=colors[0])
    plt.plot(ipv6_sorted, ipv6_y, label=f'IPv6 (mean={ipv6_mean:.2f}ms)', linewidth=2.5, color=colors[1])

    # Fill the area between curves to highlight the K-S distance
    # Find the point of maximum difference (Kolmogorov-Smirnov statistic)
    # We need to interpolate to get CDFs on the same x-axis points
    common_x = np.linspace(min(x_min, np.min(ipv4_sorted), np.min(ipv6_sorted)), 
                        max(x_max, np.max(ipv4_sorted), np.max(ipv6_sorted)), 1000)
    ipv4_cdf_interp = np.interp(common_x, ipv4_sorted, ipv4_y)
    ipv6_cdf_interp = np.interp(common_x, ipv6_sorted, ipv6_y)
    diff = np.abs(ipv4_cdf_interp - ipv6_cdf_interp)
    max_diff_idx = np.argmax(diff)
    max_diff_x = common_x[max_diff_idx]
    max_diff_ipv4_y = np.interp(max_diff_x, ipv4_sorted, ipv4_y)
    max_diff_ipv6_y = np.interp(max_diff_x, ipv6_sorted, ipv6_y)

    # Draw the K-S distance
    plt.plot([max_diff_x, max_diff_x], [max_diff_ipv4_y, max_diff_ipv6_y], 'k-', linewidth=2, label=f'K-S Distance: {ks_stat:.4f}')
    plt.plot(max_diff_x, max_diff_ipv4_y, 'ko', markersize=6)
    plt.plot(max_diff_x, max_diff_ipv6_y, 'ko', markersize=6)

    # Add annotations for the K-S distance
    mid_y = (max_diff_ipv4_y + max_diff_ipv6_y) / 2
    plt.annotate(f'K-S Distance = {ks_stat:.4f}',
                xy=(max_diff_x, mid_y), xytext=(max_diff_x + 10, mid_y),
                arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0),
                fontsize=12, ha='left', va='center')

    # Title and labels
    plt.title(f'CDF with Statistical Distance', fontsize=16)
    plt.xlabel('Round Trip Time (ms)', fontsize=14)
    plt.ylabel('Cumulative Probability', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

    # Add median (50th percentile) reference lines
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=ipv4_median, color=colors[0], linestyle='--', alpha=0.7)
    plt.axvline(x=ipv6_median, color=colors[1], linestyle='--', alpha=0.7)

    # Add annotations for median
    plt.text(ipv4_median + 2, 0.52, f'IPv4 Median:\n{ipv4_median:.1f} ms', 
            color=colors[0], fontsize=10, ha='left')
    plt.text(ipv6_median + 2, 0.48, f'IPv6 Median:\n{ipv6_median:.1f} ms', 
            color=colors[1], fontsize=10, ha='left')

    # Add explanation of statistical tests
    stat_text = (
        f"Wasserstein Distance: {w_distance:.2f} ms\n"
        f"Jensen-Shannon Distance: {js_distance:.4f}\n"
        f"Statistical Difference: {'Significant (p<0.05)' if p_value < 0.05 else 'Not Significant (p≥0.05)'}"
    )
    plt.text(0.02, 0.02, stat_text, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.savefig(rtt_cdf, dpi=300, bbox_inches='tight')

plot(ipv4_rtts_by_domain, ipv6_rtts_by_domain, "by_domain")
plot(ipv4_rtts_by_probe_domain, ipv6_rtts_by_probe_domain, "by_probe_domain")