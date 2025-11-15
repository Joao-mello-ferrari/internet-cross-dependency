import argparse
import glob
import openchord_modified_latencies as ocd
import numpy as np
from pathlib import Path

# ===================
# Argument Parser
# ===================
def parse_arguments():
    parser = argparse.ArgumentParser(description='Create country dependency heatmap and chord visualizations')
    parser.add_argument("--save", action="store_true", help="Save the figures")
    parser.add_argument("--show-plot", action="store_true", help="Show the plots interactively")
    parser.add_argument("--class", type=int, choices=[1, 2, 3, 4], help="Filter analysis by website class (1=Critical Services, 2=News, 3=General Digital Services, 4=Entertainment)")
    parser.add_argument("--code", type=str, help="Filter analysis by website class (1=Critical Services, 2=News, 3=General Digital Services, 4=Entertainment)")
    return parser.parse_args()

def get_cdn_data(class_filter=None, code=None):
    import json
    from collections import defaultdict
    from src.steps.analysis.helpers import get_class_mapping
    
    countries = {}
    with open("countries.json") as f:
        countries = json.load(f)
    
    with open("classified_websites.json") as f:
        domain_to_class = json.load(f)

    # Used to store all latencies for destinations served by each CDN
    total_domains = 0
    latency_for_cdns = defaultdict(list)
    global_website_cdn_map = defaultdict(set)
    all_cdns = set()

    # First pass: collect all CDN mappings
    for country_code in countries.keys():
        if country_code not in ["jp"]: continue
        vpns_experiment = glob.glob("results/" + country_code + "/locality/*")

        for vpn_experiment in vpns_experiment:
            cdn_file = Path(vpn_experiment) / "cdn.json"
            if not cdn_file.exists():
                continue
                
            with open(cdn_file) as f:
                cdn_data = json.load(f)
        
            for domain, cdn_string in cdn_data.items():
                # Apply class filter if provided
                if class_filter is not None and domain_to_class is not None:
                    domain_class_name = domain_to_class.get(domain)
                    if domain_class_name is None:
                        continue
                    class_mapping = get_class_mapping()
                    domain_class_num = class_mapping.get(domain_class_name)
                    if domain_class_num != class_filter:
                        continue
                    
                if not cdn_string: 
                    continue
                    
                cdn_providers_raw = set(cdn_string.replace("'", "").split(", "))
                providers_cleaned = set()
                for provider in map(str.lower, cdn_providers_raw):
                    if provider == "aws (not cdn)":
                        continue
                    if provider == "amazon aws":
                        provider = "cloudfront"
                    providers_cleaned.add(provider)

                global_website_cdn_map[domain] = global_website_cdn_map[domain].union(providers_cleaned)
                all_cdns = all_cdns.union(providers_cleaned)

    # Second pass: collect latencies for each CDN
    from src.steps.analysis.latency.data_parse_helpers import setup_paths, get_rtts
    
    for country_code in countries.keys():
        # We can use this filter to only include specific origin (not host) countries
        countries_to_include = []
        bypass_filter =  len(countries_to_include) == 0
        country_code_filter = country_code in countries_to_include
        if not (bypass_filter or country_code_filter): continue

        for cdn in all_cdns:
            try:
                base_path, latency_ipv4_input, latency_ipv6_input, fail_ipv6_input = setup_paths(country_code)
                _, _, ipv4_by_probe, _, _, domains_count = get_rtts(
                    latency_ipv4_input,
                    latency_ipv6_input,
                    fail_ipv6_input,
                    require_both_procols_results=False,
                    filter_mapper = global_website_cdn_map,
                    filter_key = cdn
                )
                latency_for_cdns[cdn].extend(ipv4_by_probe)
                total_domains += domains_count
            except Exception as e:
                print(f"Error processing {country_code} for CDN {cdn}: {e}")
                continue
    
    return dict(latency_for_cdns)


def get_country_data(class_filter=None, code = None):
    import json
    from collections import defaultdict
    from src.steps.analysis.helpers import get_class_mapping
    
    countries = {}
    with open("countries.json") as f:
        countries = json.load(f)
    
    with open("classified_websites.json") as f:
        domain_to_class = json.load(f)
    
    # Used to store all latencies for destinations served by each CDN
    total_domains = 0
    latency_for_countries = defaultdict(list)
    global_website_country_map = defaultdict(set)
    all_countries = set()

    # First pass: collect all CDN mappings
    for country_code in countries.keys():
        if country_code not in ["jp"]: continue
        vpns_experiment = glob.glob("results/" + country_code + "/locality/*")

        for vpn_experiment in vpns_experiment:
            cdn_file = Path(vpn_experiment) / "cdn.json"
            location_file = Path(vpn_experiment) / "location.json"
            locedge_file = Path(vpn_experiment) / "locedge.json"
            if not cdn_file.exists() and (not location_file.exists() or not locedge_file.exists()):
                continue
                
            with open(cdn_file) as f:
                cdn_data = json.load(f)
            with open(location_file) as f:
                location_data = json.load(f)
            with open(locedge_file) as f:
                locedge_data = json.load(f)
        
            for domain, cdn_string in cdn_data.items():
                # Apply class filter if provided
                if class_filter is not None and domain_to_class is not None:
                    domain_class_name = domain_to_class.get(domain)
                    if domain_class_name is None:
                        continue
                    class_mapping = get_class_mapping()
                    domain_class_num = class_mapping.get(domain_class_name)
                    if domain_class_num != class_filter:
                        continue

                # Not served via CDN. Skipping
                if not cdn_string: 
                    continue
                
                locality = location_data.get(domain)
                locedge = locedge_data.get(domain, {}).get("contentLocality")

                final_location = None
                if locality and "- anycast" not in locality:
                    final_location = locality
                elif locedge:
                     final_location = locedge

                if final_location:
                    global_website_country_map[domain] = global_website_country_map[domain].union({final_location})
                    all_countries = all_countries.union({final_location})    

    # Second pass: collect latencies for each CDN
    from src.steps.analysis.latency.data_parse_helpers import setup_paths, get_rtts
    from src.steps.analysis.helpers import country_names
    
    for country_code in countries.keys():
        # We can use this filter to only include specific origin (not host) countries
        countries_to_include = []
        bypass_filter =  len(countries_to_include) == 0
        country_code_filter = country_code in countries_to_include
        if not (bypass_filter or country_code_filter): continue

        for country in all_countries:
            try:
                base_path, latency_ipv4_input, latency_ipv6_input, fail_ipv6_input = setup_paths(country_code)
                _, _, ipv4_by_probe, _, _, domains_count = get_rtts(
                    latency_ipv4_input,
                    latency_ipv6_input,
                    fail_ipv6_input,
                    require_both_procols_results=False,
                    filter_mapper = global_website_country_map,
                    filter_key = country
                )
                country_name = country_names.get(country, country)
                if country_name != "unknown":
                    latency_for_countries[country_name].extend(ipv4_by_probe)
                    total_domains += domains_count
            except Exception as e:
                print(f"Error processing {country_code} for country {country}: {e}")
                continue

    return dict(latency_for_countries)

def get_country_latency_matrix():
    """
    Get country latency data as an n x n matrix where:
    - n = number of countries with measurement data
    - matrix[i,j] = median latency from country i to websites hosted in country j
    
    Returns:
        matrix: n x n numpy array of median latencies (in ms)
        country_labels: list of country names corresponding to matrix indices
    """
    import json
    import numpy as np
    from collections import defaultdict
    from src.steps.analysis.helpers import country_names, sort_countries_by_continent
    
    countries = {}
    with open("countries.json") as f:
        countries = json.load(f)
    
    # Store latencies organized by (source_country, target_country)
    latency_matrix_data = defaultdict(lambda: defaultdict(list))
    all_target_countries = set()
    source_countries = list(countries.keys())
    
    print(f"Processing {len(source_countries)} source countries...")
    
    for source_country_code in source_countries:
        print(f"Processing source country: {source_country_code}")
        vpns_experiment = glob.glob("results/" + source_country_code + "/locality/*")

        for vpn_experiment in vpns_experiment:
            cdn_file = Path(vpn_experiment) / "cdn.json"
            location_file = Path(vpn_experiment) / "location.json"
            locedge_file = Path(vpn_experiment) / "locedge.json"
            
            if not (cdn_file.exists() and location_file.exists() and locedge_file.exists()):
                continue
                
            # Load the data files
            with open(cdn_file) as f:
                cdn_data = json.load(f)
            with open(location_file) as f:
                location_data = json.load(f)
            with open(locedge_file) as f:
                locedge_data = json.load(f)
            
            # Create mapping from domain to target country
            domain_to_target_country = {}
            for domain in cdn_data.keys():
                if not cdn_data[domain]:  # Skip if not served via CDN
                    continue
                
                # Determine the target country for this domain
                locality = location_data.get(domain)
                locedge = locedge_data.get(domain, {}).get("contentLocality")
                
                final_location = None
                if locality and "- anycast" not in locality:
                    final_location = locality
                elif locedge:
                    final_location = locedge
                
                if final_location:
                    domain_to_target_country[domain] = final_location
                    all_target_countries.add(final_location)
            
            # Get latencies for domains in this experiment
            if domain_to_target_country:
                from src.steps.analysis.latency.data_parse_helpers import setup_paths, get_rtts
                
                try:
                    base_path, latency_ipv4_input, latency_ipv6_input, fail_ipv6_input = setup_paths(source_country_code)
                    
                    # Process each target country separately
                    for target_country in set(domain_to_target_country.values()):
                        # Create filter for domains going to this target country
                        target_domains = {domain for domain, country in domain_to_target_country.items() 
                                        if country == target_country}
                        
                        if target_domains and len(target_domains) > 5:
                            filter_mapper = {domain: {target_country} for domain in target_domains}
                            
                            _, _, ipv4_by_probe, _, _, _ = get_rtts(
                                latency_ipv4_input,
                                latency_ipv6_input,
                                fail_ipv6_input,
                                require_both_procols_results=False,
                                filter_mapper=filter_mapper,
                                filter_key=target_country
                            )
                            
                            # Store latencies for this source->target pair
                            latency_matrix_data[source_country_code][target_country].extend(ipv4_by_probe)
                
                except Exception as e:
                    print(f"Error processing latencies for {source_country_code}: {e}")
                    continue
    
    # Filter to countries that appear in both source and target
    valid_countries = []
    for country_code in source_countries:
        country_name = country_names.get(country_code, country_code)
        # Check if this country has any latency data (either as source or target)
        has_source_data = len(latency_matrix_data[country_code]) > 0
        has_target_data = any(country_code in target_countries for target_countries in latency_matrix_data.values())
        

        # We can use this filter to only include specific origin (not host) countries
        countries_to_include = []
        bypass_filter =  len(countries_to_include) == 0
        country_code_filter = country_code in countries_to_include
        if (has_source_data or has_target_data) and (bypass_filter or country_code_filter):
            valid_countries.append(country_code)
    
    print(f"Found {len(valid_countries)} countries with latency data")
    
    # Create country labels (convert codes to names)
    valid_countries = sort_countries_by_continent(valid_countries)
    country_labels = [country_names.get(code, code) for code in valid_countries]
    
    # Create the matrix
    n_countries = len(valid_countries)
    matrix = np.full((n_countries, n_countries), np.nan)
    
    # Coefficient of Variation (CV) threshold
    # CV = (std_dev / mean) * 100
    # CV < 40% indicates acceptable consistency in latency measurements
    # This threshold is chosen because:
    # - CV < 30%: Very consistent (low variability)
    # - CV 30-40%: Moderately consistent (acceptable for network measurements)
    # - CV 40-50%: High variability (questionable reliability)
    # - CV > 50%: Very high variability (unreliable measurements)
    CV_THRESHOLD = 40.0  # Percentage
    
    for i, source_code in enumerate(valid_countries):
        for j, target_code in enumerate(valid_countries):
            if target_code in latency_matrix_data[source_code]:
                latencies = latency_matrix_data[source_code][target_code]
                if latencies and len(latencies) >= 5:  # Need at least 5 measurements
                    mean_latency = np.mean(latencies)
                    std_latency = np.std(latencies)

                    # Calculate Coefficient of Variation (CV)
                    if mean_latency > 0:
                        cv = (std_latency / mean_latency) * 100
                        
                        if len(latencies) < 100:
                            coeff = 1
                        elif len(latencies) < 300:
                            coeff = 1.3
                        elif len(latencies) < 500:
                            coeff = 1.6
                        elif len(latencies) < 1000:
                            coeff = 1.9
                        elif len(latencies) < 2000:
                            coeff = 2.2
                        else:
                            coeff = 2.5

                        # Only add to matrix if CV is below threshold (low dispersion)
                        if cv <= (CV_THRESHOLD * coeff):
                            matrix[i, j] = np.median(latencies)
                            print(f"{country_names.get(source_code, source_code)} -> {country_names.get(target_code, target_code)}: "
                                  f"{len(latencies)} measurements, median: {matrix[i,j]:.1f}ms, CV: {cv:.1f}% ✓")
                        else:
                            print(f"{country_names.get(source_code, source_code)} -> {country_names.get(target_code, target_code)}: "
                                  f"{len(latencies)} measurements, median: {np.median(latencies):.1f}ms, CV: {cv:.1f}% ✗ (high dispersion, excluded)")
    
    return matrix, country_labels

def create_latency_boxplots(data, title="Latency Distribution", save_path=None, data_type="provider", 
                          min_count_threshold=200, primary_color="#690398", others_color="#E9E911"):
    """
    Create box plots showing latency distribution for each provider (CDN/Country).
    
    Args:
        data: dict of {provider_name: [latency_values]}
        title: plot title
        save_path: path to save the figure (optional)
        data_type: "provider" for CDN, "country" for countries, or custom label
        min_count_threshold: minimum measurements to show individually (others grouped)
        primary_color: color for individual providers
        others_color: color for grouped "Others" category
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Filter out providers with no data and outliers
    def remove_outliers(latencies: list) -> list:
        if len(latencies) < 10:
            return latencies
        p5, p95 = np.percentile(latencies, [5, 95])
        return [x for x in latencies if p5 <= x <= p95]

    filtered_data = {provider: remove_outliers(latencies) for provider, latencies in data.items() 
                    if latencies and len(latencies) > 0}
    
    if not filtered_data:
        print(f"No {data_type} data available for plotting")
        return
    
    # Calculate statistics for sorting and grouping
    provider_stats = []
    others_latencies = []
    others_count = 0
    
    for provider, latencies in filtered_data.items():
        median_latency = np.median(latencies)
        count = len(latencies)
        
        if count < min_count_threshold:
            # Group into "Others"
            others_latencies.extend(latencies)
            others_count += count
        else:
            provider_stats.append((provider, latencies, median_latency, count))
    
    # Sort by median latency (ascending - lower is better)
    provider_stats.sort(key=lambda x: x[2])
    
    # Prepare data for plotting
    provider_names = []
    latency_data = []
    provider_counts = []
    
    # Add individual providers with >= min_count_threshold measurements
    for provider, latencies, median_lat, count in provider_stats:
        # Format name (capitalize and truncate if too long)
        if data_type.lower() == "host countries":
            formatted_name = provider[:15] + ("..." if len(provider) > 15 else "")
        else:
            formatted_name = provider.capitalize()[:12] + ("..." if len(provider) > 12 else "")
        
        provider_names.append(formatted_name)
        latency_data.append(latencies)
        provider_counts.append(count)
    
    # Add "Others" group if it has data
    if others_latencies:
        provider_names.append("Others")
        latency_data.append(remove_outliers(others_latencies))
        provider_counts.append(others_count)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(max(12, len(provider_names) * 0.8), 6))
    
    # Create box plots
    box_plot = ax.boxplot(latency_data, labels=provider_names, patch_artist=True)
    
    # Color the boxes
    colors = []
    for i, name in enumerate(provider_names):
        if name == "Others":
            colors.append(others_color)
        else:
            colors.append(primary_color)
    
    # Apply colors to boxes
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize box plot elements
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_plot[element], color='black')
    
    # Customize the plot
    #ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax.set_ylabel('Median Latency (ms)', fontsize=18)
    ax.tick_params(axis='x', rotation=45, labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add count labels above each box
    y_max = max([max(latencies) for latencies in latency_data])
    for i, count in enumerate(provider_counts):
        ax.text(i+1, y_max * 1.05, f'n={count}', ha='center', va='bottom', fontsize=11)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=primary_color, alpha=0.7, label=data_type)]
    if others_latencies:
        legend_elements.append(Patch(facecolor=others_color, alpha=0.7, label=f'Grouped {data_type}'))
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{data_type.title()} latency box plots saved to: {save_path}")
    
    plt.show()
    
    return provider_names, [np.median(data) for data in latency_data]

def create_probability_density_plots(data, title="Latency Probability Density", save_path=None, 
                                   data_type="provider", top_n=5):
    """
    Create probability density plots for the top N providers with most measurement points.
    
    Args:
        data: dict of {provider_name: [latency_values]}
        title: plot title
        save_path: path to save the figure (optional)
        data_type: "provider" for CDN, "country" for countries, or custom label
        top_n: number of top providers to show
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde
    
    # Filter out providers with no data and calculate counts
    filtered_data = {provider: latencies for provider, latencies in data.items() 
                    if latencies and len(latencies) > 0}
    
    if not filtered_data:
        print(f"No {data_type} data available for plotting")
        return
    
    # Get top N providers by measurement count
    provider_counts = [(provider, len(latencies), latencies) for provider, latencies in filtered_data.items()]
    provider_counts.sort(key=lambda x: x[1], reverse=True)  # Sort by count (descending)
    top_providers = provider_counts[:top_n]
    
    print(f"Top {top_n} {data_type}s by measurement count:")
    for provider, count, _ in top_providers:
        print(f"  {provider}: {count:,} measurements")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors for the top providers
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot density for each provider
    for i, (provider, count, latencies) in enumerate(top_providers):
        # Remove extreme outliers for better visualization (keep 90% of data)
        latencies_clean = np.array(latencies)
        # p5, p95 = np.percentile(latencies_clean, [5, 95])
        # latencies_filtered = latencies_clean[(latencies_clean >= p5) & (latencies_clean <= p95)]
        latencies_filtered = latencies_clean  # Using all data for KDE
        
        if len(latencies_filtered) < 10:  # Need enough points for KDE
            continue
            
        # Create probability density using kernel density estimation
        kde = gaussian_kde(latencies_filtered)
        
        # Create x-axis range for smooth curve
        x_min, x_max = np.min(latencies_filtered), np.max(latencies_filtered)
        x_range = np.linspace(x_min, x_max, 300)
        
        # Calculate density
        density = kde(x_range)
        
        # Format provider name
        if data_type.lower() == "country":
            formatted_name = provider[:15] + ("..." if len(provider) > 15 else "")
        else:
            formatted_name = provider.capitalize()
            if len(formatted_name) > 15:
                formatted_name = formatted_name[:12] + "..."
        
        # Plot the density curve
        color = colors[i % len(colors)]
        ax.plot(x_range, density, color=color, linewidth=2.5, 
               label=f'{formatted_name} (n={count:,})', alpha=0.8)
        
        # Fill under the curve with transparency
        ax.fill_between(x_range, density, alpha=0.3, color=color)
    
    # Customize the plot
    ax.set_xlabel('Latency (ms)', fontsize=18)
    ax.set_ylabel('Probability Density', fontsize=18)
    #ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # Set reasonable x-axis limits (remove extreme outliers for better view)
    all_latencies = []
    for _, _, latencies in top_providers:
        all_latencies.extend(latencies)
    if all_latencies:
        p1, p99 = np.percentile(all_latencies, [1, 99])
        ax.set_xlim(max(0, p1), p99)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{data_type.title()} probability density plots saved to: {save_path}")
    
    plt.show()
    
    return [provider for provider, _, _ in top_providers]

def create_country_latency_heatmap(matrix, country_labels, title="Country-to-Country Median Latency", save_path=None, show_plot=True):
    """
    Create a heatmap showing median latencies between countries.
    
    Args:
        matrix: n x n numpy array of median latencies (in ms)
        country_labels: list of country names corresponding to matrix indices
        title: plot title
        save_path: path to save the figure (optional)
        show_plot: whether to show the plot interactively
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Set up the plot
    figsize = (max(12, len(country_labels) * 0.8), max(10, len(country_labels) * 0.7))
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a mask for NaN values to make them transparent
    mask_nan = np.isnan(matrix)

    # Formatted country names
    formatted_country_labels = []
    for label in country_labels:
        if len(label) > 12:
            formatted_country_labels.append(label[:11] + ".")
        else:
            formatted_country_labels.append(label)

    # Create the heatmap
    sns.heatmap(
        matrix,
        xticklabels=formatted_country_labels,
        yticklabels=country_labels,
        annot=True,
        fmt='.0f',
        cmap='viridis',  # Reversed viridis (yellow=low latency, purple=high latency)
        cbar_kws={'label': 'Median Latency (ms)'},
        square=False,
        linewidths=1,
        linecolor='white',
        mask=mask_nan,  # This makes NaN values transparent
        ax=ax
    )
    
    # Increase colorbar label font size
    cbar = ax.collections[0].colorbar
    cbar.set_label('CDN Provider Concentration (%)', fontsize=15)

    
    # Style the plot
    #ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Host Countries', fontsize=15)
    ax.set_ylabel('Measurement Origins', fontsize=15)
    
    # Rotate labels for better readability
    ax.tick_params(axis='x', which='major', labelsize=14, top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.tick_params(axis='y', which='major', labelsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='left')
    
    # Color-code provider labels based on type
    x_labels = ax.get_xticklabels()
    for i, label in enumerate(x_labels):
        label.set_color('blue')

    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Country latency heatmap saved to: {save_path}")
    
    # Show the plot
    if show_plot:
        plt.show()
    
    return matrix


def create_country_latency_chord(matrix, country_labels, save_path=None, show_plot=True):
    """
    Create a dependency chord diagram visualization.
    
    Args:
        matrix: n x n matrix of dependencies between countries
        country_labels: list of country names/codes
        save_path: path to save the figure (optional)
    """
    
    # Create a copy to avoid modifying the original matrix
    matrix = matrix.copy()
    
    # Remove rows and columns only if BOTH the row AND column at the same index are entirely NaN
    indices_to_keep = []
    for i in range(matrix.shape[0]):
        row_all_nan = np.all(np.isnan(matrix[i, :]))
        col_all_nan = np.all(np.isnan(matrix[:, i]))
        
        # Keep this index if either the row OR the column has at least one non-NaN value
        if not (row_all_nan and col_all_nan):
            indices_to_keep.append(i)
    
    matrix = matrix[indices_to_keep][:, indices_to_keep]
    country_labels = [country_labels[i] for i in indices_to_keep]
    matrix = np.nan_to_num(matrix, nan=0.0)
    print(matrix)

    # Make matrix symmetric by adding small values to zero cells in transpose positions
    for n in range(matrix.shape[0]):
        for m in range(matrix.shape[1]):
            # If cell (n,m) has a value and cell (m,n) is zero, add small value to (m,n)
            if matrix[n, m] != 0 and matrix[m, n] == 0:
                matrix[m, n] = matrix[n, m] * 0.01

    # Calculate column sums and apply cubic root transformation for gentle scaling
    col_sums = np.sum(matrix, axis=0)
    col_sums_cbrt = np.sqrt(col_sums + 1e-10)  # Add tiny value for numerical stability

    # Apply scaling
    matrix_scaled = matrix * col_sums_cbrt[:, np.newaxis]

    # Create and display the chord diagram
    print(f"Creating chord diagram with {len(country_labels)} countries...")
    print(f"Original column sums: {col_sums}")
    print(f"Scaled column sums: {col_sums_cbrt}")
    
    fig = ocd.Chord(matrix_scaled, country_labels, scale=list(col_sums_cbrt))
    if show_plot:
        fig.show()
    
    # Save if path provided
    if save_path:
        fig.save_svg(save_path)
        print(f"Chord diagram saved to: {save_path}")
    
    return matrix_scaled


def main():    
    """
    Main function to create latency visualizations for both CDNs and countries
    """
    import numpy as np
    
    args = parse_arguments()
    _class = getattr(args, 'class', None)
    code = args.code
    class_save_appendix = f"_class{_class}" if _class is not None else ""
    code_save_appendix = f"_{code}" if code is not None else "" 
    
    # Get CDN data
    print("Fetching CDN data...")
    cdn_data = get_cdn_data(_class, code)
    print(f"Found {len(cdn_data)} CDN providers...")
    
    # Print CDN statistics
    for cdn, latencies in cdn_data.items():
        if latencies:
            print(f"  {cdn}: {len(latencies)} measurements, median: {np.median(latencies):.1f}ms")
    
    # Create CDN visualizations
    print("\n" + "="*60)
    print("Creating CDN latency box plots...")
    create_latency_boxplots(
        cdn_data, 
        title="CDN Latency Distribution Across All Countries",
        data_type="CDN Providers",
        min_count_threshold=500,
        save_path=f"results/cdn_latency_boxplots{class_save_appendix}{code_save_appendix}.png" if args.save else None
    )
    
    print("\n" + "="*60)
    print("Creating CDN probability density plots...")
    create_probability_density_plots(
        cdn_data,
        title="CDN Latency Probability Density - Top 5 Providers",
        data_type="CDN Provider",
        save_path=f"results/cdn_probability_density{class_save_appendix}{code_save_appendix}.png" if args.save else None
    )
    
    # Get country data
    print("\n" + "="*60)
    print("Fetching country data...")
    country_data = get_country_data(_class, code)
    print(f"Found {len(country_data)} countries...")
    
    # Print country statistics
    for country, latencies in country_data.items():
        if latencies:
            print(f"  {country}: {len(latencies)} measurements, median: {np.median(latencies):.1f}ms")
    
    # Create country visualizations
    print("\n" + "="*60)
    print("Creating country latency box plots...")
    create_latency_boxplots(
        country_data, 
        title="Country Latency Distribution Across All Measurements",
        data_type="Host Countries",
        min_count_threshold=1000,  # Lower threshold for countries
        #primary_color="#2ca02c",  # Green for countries
        #others_color="#ff7f0e",   # Orange for others
        save_path=f"results/country_latency_boxplots{class_save_appendix}{code_save_appendix}.png" if args.save else None
    )
    
    print("\n" + "="*60)
    print("Creating country probability density plots...")
    create_probability_density_plots(
        country_data,
        title="Country Latency Probability Density - Top 5 Countries",
        data_type="Country",
        save_path=f"results/country_probability_density{class_save_appendix}{code_save_appendix}.png" if args.save else None,
        top_n=3
    )
    
    # Get country latency matrix and create heatmap
    print("\n" + "="*60)
    print("Creating country-to-country latency matrix...")
    latency_matrix, matrix_country_labels = get_country_latency_matrix()
    
    if latency_matrix is not None and len(matrix_country_labels) > 0:
        print(f"Created {latency_matrix.shape[0]}x{latency_matrix.shape[1]} latency matrix")
        
        print("\n" + "="*60)
        print("Creating country latency heatmap...")
        create_country_latency_heatmap(
            latency_matrix,
            matrix_country_labels,
            title="Country-to-Country Median Latency Heatmap",
            save_path=f"results/country_latency_heatmap{class_save_appendix}.png" if args.save else None,
            show_plot=args.show_plot if hasattr(args, 'show_plot') else True
        )

        create_country_latency_chord(
            latency_matrix,
            matrix_country_labels,
            #title="Country-to-Country Median Latency Chord Diagram",
            save_path=f"results/country_latency_chord{class_save_appendix}.svg" if args.save else None,
            show_plot=args.show_plot if hasattr(args, 'show_plot') else True
        )


    else:
        print("No country latency matrix data available")


if __name__ == "__main__":
    main()


    

        