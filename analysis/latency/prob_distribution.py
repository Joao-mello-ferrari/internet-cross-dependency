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

with open(fail_ipv6_input, "r") as f:
    fail_ipv6_route = json.load(f)



# PLOT GENERATION
def plot(ipv4_rtts, ipv6_rtts, subpath):
    probability_distribution = base_path / "results" / subpath / f"probability_distribution.png"
    
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


    # Create distribution comparison plot with statistical distances
    plt.figure(figsize=(12, 8))

    # Create subplots
    gs = GridSpec(2, 1, height_ratios=[3, 1], figure=plt.gcf())
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # Plot the smooth probability distributions
    ax1.plot(x_range, ipv4_density, linewidth=2.5, label=f'IPv4 (mean={ipv4_mean:.2f}ms)', color=colors[0])
    ax1.plot(x_range, ipv6_density, linewidth=2.5, label=f'IPv6 (mean={ipv6_mean:.2f}ms)', color=colors[1])

    # Add vertical lines for means
    ax1.axvline(x=ipv4_mean, color=colors[0], linestyle='--', alpha=0.7)
    ax1.axvline(x=ipv6_mean, color=colors[1], linestyle='--', alpha=0.7)

    # Fill the area to highlight the distribution difference
    ax1.fill_between(x_range, ipv4_density, ipv6_density, where=(ipv4_density > ipv6_density), 
                    interpolate=True, color=colors[0], alpha=0.3, label='IPv4 better')
    ax1.fill_between(x_range, ipv4_density, ipv6_density, where=(ipv6_density > ipv4_density), 
                    interpolate=True, color=colors[1], alpha=0.3, label='IPv6 better')

    # Add title, labels, and legend
    ax1.set_title(f'IPv4 vs IPv6 RTT Probability Distributions', fontsize=16)
    ax1.set_xlabel('Round Trip Time (ms)', fontsize=14)
    ax1.set_ylabel('Probability Density', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12, loc='upper right')

    # Add statistical information
    stat_text = (
        f"Statistical Significance: {'Yes (p<0.05)' if p_value < 0.05 else 'No (p≥0.05)'}\n"
        f"Jensen-Shannon Distance: {js_distance:.4f}\n"
        f"Wasserstein Distance: {w_distance:.2f} ms\n"
        f"Kolmogorov-Smirnov Stat: {ks_stat:.4f}"
    )
    ax1.text(0.02, 0.98, stat_text, transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    # Add histogram of actual data points (at the bottom)
    upper_bound = max(np.percentile(ipv4_rtts, 99), np.percentile(ipv6_rtts, 99)) * 1.2
    lower_bound = min(np.min(ipv4_rtts), np.min(ipv6_rtts))
    bin_width = (upper_bound - lower_bound) / 40  # Adjust for appropriate bin width
    bins = np.arange(lower_bound, upper_bound + bin_width, bin_width)
    ax2.hist(ipv4_rtts, bins=bins, alpha=0.7, color=colors[0], label='IPv4')
    ax2.hist(ipv6_rtts, bins=bins, alpha=0.7, color=colors[1], label='IPv6')
    ax2.set_xlabel('Round Trip Time (ms)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(probability_distribution, dpi=300, bbox_inches='tight')


plot(ipv4_rtts_by_domain, ipv6_rtts_by_domain, "by_domain")
plot(ipv4_rtts_by_probe_domain, ipv6_rtts_by_probe_domain, "by_probe_domain")
