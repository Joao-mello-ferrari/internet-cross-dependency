import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy import stats
from scipy.spatial.distance import jensenshannon
from matplotlib.gridspec import GridSpec
from src.steps.analysis.latency.data_parse_helpers import setup_paths, get_rtts

parser = argparse.ArgumentParser(description="Generate RTT plots comparing IPv4 and IPv6 latency normal distributions.")
parser.add_argument("--country", type=str, required=True, help="Country label")
parser.add_argument("--code", type=str, required=True, help="Country code used for folder path")
parser.add_argument("--save", action='store_true', help="Save the generated figures")
args = parser.parse_args()


# PLOT GENERATION
def plot(ipv4_rtts, ipv6_rtts, label, subpath):
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
    ax1.set_title(f'IPv4 vs IPv6 RTT Probability Distributions ({args.country}, {label})', fontsize=16)
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
    #ax1.text(0.02, 0.98, stat_text, transform=ax1.transAxes, fontsize=12,
    #        verticalalignment='top', horizontalalignment='left',
    #        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

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
    if args.save:
        plt.savefig(probability_distribution, dpi=300)
    else:
        plt.show()
        plt.close()

# --- Load Latency Data ---
base_path, latency_ipv4_input, latency_ipv6_input, fail_ipv6_input = setup_paths(args.code)
ipv4_by_domain, ipv6_by_domain, ipv4_by_probe, ipv6_by_probe, fail_ipv6_route, _ = get_rtts(
    latency_ipv4_input,
    latency_ipv6_input,
    fail_ipv6_input,
)
fail_count = len(fail_ipv6_route)

# --- Generate Plots ---
plot(ipv4_by_domain, ipv6_by_domain, "aggregated by domain", "latency/by_domain")
plot(ipv4_by_probe, ipv6_by_probe, "aggregated by probe", "latency/by_probe")

