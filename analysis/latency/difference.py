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
from scipy.signal import savgol_filter

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
        
        if len(filtered_ipv4) == 0 or len(filtered_ipv6) == 0:  
            continue
        
        ipv4_results.append(median(filtered_ipv4))
        ipv6_results.append(median(filtered_ipv6))

    return ipv4_results, ipv6_results, ipv4_error, ipv6_error

def _filter(ipv4_domain_probes, ipv6_domain_probes):
    ipv4_results, ipv6_results = [], []
    for probe_ipv4, probe_ipv6 in zip(ipv4_domain_probes, ipv6_domain_probes):
        filtered_ipv4 = list(filter(lambda result: result != None and result != 0, probe_ipv4))
        filtered_ipv6 = list(filter(lambda result: result != None and result != 0, probe_ipv6))

        if len(filtered_ipv4) == 0 or len(filtered_ipv6) == 0:  
            continue
        
        ipv4_results.append(median(filtered_ipv4))
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
    rtts_difference_histogram_output = base_path / "results" / subpath / f"rtts_difference_histogram.png"
    sorted_rtts_difference_output = base_path / "results" / subpath / f"sorted_rtts_difference.png"
    rtts_by_difference_output = base_path / "results" / subpath / f"rtts_by_difference.png"
    
    # Set the visual styles
    plt.style.use('ggplot')
    sns.set(style="whitegrid", font_scale=1.2)
    colors = ["#3498db", "#e74c3c", "#2ecc71"]  # Blue, Red, Green

    # Calculate the direct difference (positive means IPv6 is slower)
    rtt_difference = ipv6_rtts - ipv4_rtts

    # Create a DataFrame with all the information for easier manipulation
    data = pd.DataFrame({
        #'probe_id': probe_ids,
        'ipv4_rtt': ipv4_rtts,
        'ipv6_rtt': ipv6_rtts,
        'difference': rtt_difference
    })

    # Sort by the difference value
    sorted_data = data.sort_values('difference').reset_index(drop=True)

    # Basic statistics on the difference
    mean_diff = np.mean(rtt_difference)
    median_diff = np.median(rtt_difference)
    std_diff = np.std(rtt_difference)
    percent_positive = np.mean(rtt_difference > 0) * 100
    percent_negative = np.mean(rtt_difference < 0) * 100

    print(f"Mean difference (IPv6 - IPv4): {mean_diff:.2f} ms")
    print(f"Median difference: {median_diff:.2f} ms")
    print(f"Standard deviation of difference: {std_diff:.2f} ms")
    print(f"Percentage where IPv6 is slower: {percent_positive:.1f}%")
    print(f"Percentage where IPv4 is slower: {percent_negative:.1f}%")

    # Perform t-test to check if the mean difference is statistically significant
    t_stat, p_value = stats.ttest_1samp(rtt_difference, 0)
    print(f"T-test for difference != 0: t={t_stat:.4f}, p-value={p_value:.4f}")
    print(f"The difference is statistically {'significant' if p_value < 0.05 else 'not significant'} at α=0.05")

    # Create a stats text box to use in plots
    stats_text = (
        f"Mean difference: {mean_diff:.2f} ms\n"
        f"Median difference: {median_diff:.2f} ms\n"
        f"IPv6 slower: {percent_positive:.1f}% of pairs\n"
        f"IPv4 slower: {percent_negative:.1f}% of pairs\n"
        f"Statistically significant: {'Yes (p<0.05)' if p_value < 0.05 else 'No (p≥0.05)'}"
    )

    #################################################
    # PLOT 1: Histogram of Differences
    #################################################
    plt.figure(figsize=(12, 6))

    # Create histogram
    bins = np.linspace(min(rtt_difference), max(rtt_difference), 30)
    plt.hist(rtt_difference, bins=bins, alpha=0.7, color=colors[0])

    # Add reference lines
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='No difference')
    plt.axvline(x=mean_diff, color='purple', linestyle='-', alpha=0.7, label=f'Mean: {mean_diff:.2f} ms')
    plt.axvline(x=median_diff, color='green', linestyle='-', alpha=0.7, label=f'Median: {median_diff:.2f} ms')

    # Add annotations
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    # Customize the histogram
    plt.title('Distribution of RTT Differences (IPv6 - IPv4)', fontsize=16)
    plt.xlabel('RTT Difference (IPv6 - IPv4) in ms', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(rtts_difference_histogram_output, dpi=300, bbox_inches='tight')

    #################################################
    # PLOT 2: Sorted Difference Visualization
    #################################################
    plt.figure(figsize=(14, 8))

    # Get sorted data and x-axis
    x_sorted = np.arange(len(sorted_data))

    # Plot raw differences
    plt.scatter(x_sorted, sorted_data['difference'], alpha=0.5, color=colors[0], label='RTT Differences (sorted)')

    # Apply smoothing to sorted data
    window_size = min(15, len(sorted_data) - 1 if len(sorted_data) % 2 == 0 else len(sorted_data) - 2)
    if window_size >= 4:
        sorted_smooth = savgol_filter(sorted_data['difference'], window_size, 3)
        plt.plot(x_sorted, sorted_smooth, color=colors[1], linewidth=3, 
                label=f'Smoothed trend (Savitzky-Golay)')

    # Reference lines
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='No difference line')
    plt.axhline(y=mean_diff, color='purple', linestyle='-', alpha=0.7, label=f'Mean difference: {mean_diff:.2f} ms')

    # Add a confidence interval band (±1 standard deviation)
    plt.fill_between(x_sorted, mean_diff - std_diff, mean_diff + std_diff, 
                    color='purple', alpha=0.2, label=f'±1 std dev ({std_diff:.2f} ms)')

    # Customize the plot
    plt.title('Sorted Difference Between IPv6 and IPv4 RTT', fontsize=16)
    plt.xlabel('Measurements (sorted by difference value)', fontsize=14)
    plt.ylabel('RTT Difference (IPv6 - IPv4) in ms', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper left')

    # Add annotations
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.savefig(sorted_rtts_difference_output, dpi=300, bbox_inches='tight')

    #################################################
    # PLOT 3: Absolute RTT Values Ordered by Difference
    #################################################
    plt.figure(figsize=(14, 8))

    # Plot the absolute RTT values, using the same order as the sorted difference plot
    plt.plot(x_sorted, sorted_data['ipv4_rtt'], 'o-', color=colors[0], alpha=0.7, label='IPv4 RTT')
    plt.plot(x_sorted, sorted_data['ipv6_rtt'], 'o-', color=colors[1], alpha=0.7, label='IPv6 RTT')

    # Fill the area between the lines to highlight the difference
    plt.fill_between(x_sorted, sorted_data['ipv4_rtt'], sorted_data['ipv6_rtt'], 
                    where=(sorted_data['ipv6_rtt'] > sorted_data['ipv4_rtt']),
                    color=colors[1], alpha=0.3, label='IPv6 slower')
    plt.fill_between(x_sorted, sorted_data['ipv4_rtt'], sorted_data['ipv6_rtt'], 
                    where=(sorted_data['ipv4_rtt'] >= sorted_data['ipv6_rtt']),
                    color=colors[0], alpha=0.3, label='IPv4 slower')

    # Add vertical separator where the difference crosses zero
    crossover_idx = sorted_data['difference'].ge(0).idxmax()
    plt.axvline(x=crossover_idx, color='black', linestyle='--', alpha=0.7, 
            label=f'Crossover point ({crossover_idx}/{len(sorted_data)} measurements)')

    # Customize the plot
    plt.title('Absolute RTT Values (Ordered by Difference)', fontsize=16)
    plt.xlabel('Measurements (sorted by difference)', fontsize=14)
    plt.ylabel('Round Trip Time (ms)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper left')

    # Add explanatory text
    plt.text(0.02, 0.98, 
            f"IPv4 better | IPv6 better\n"
            f"Left of line: {crossover_idx} measurements ({crossover_idx/len(sorted_data)*100:.1f}%)\n"
            f"Right of line: {len(sorted_data)-crossover_idx} measurements ({(1-crossover_idx/len(sorted_data))*100:.1f}%)",
            transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.savefig(rtts_by_difference_output, dpi=300, bbox_inches='tight')

    print("\nVisualization completed! The following files have been generated:")
    print("1. difference_histogram.png - Histogram showing distribution of RTT differences")
    print("2. sorted_difference_plot.png - Shows sorted differences with trend line")
    print("3. absolute_rtts_by_difference.png - Shows absolute RTT values ordered by difference")

plot(ipv4_rtts_by_domain, ipv6_rtts_by_domain, "by_domain")
plot(ipv4_rtts_by_probe_domain, ipv6_rtts_by_probe_domain, "by_probe_domain")