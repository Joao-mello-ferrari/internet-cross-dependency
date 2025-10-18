import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from pathlib import Path
from collections import defaultdict

from src.steps.analysis.helpers import (
    get_all_country_codes, convert_codes_to_names, sort_countries_by_continent,
    get_continent_mapping, process_experiment_cdn, normalize_matrix, find_experiment_files,
    load_classified_websites, get_class_mapping
)

# ===================
# Argument Parser
# ===================
def parse_arguments():
    parser = argparse.ArgumentParser(description='Create CDN provider dependency heatmap visualizations')
    parser.add_argument("--country", type=str, help="Country label (for single country analysis)")
    parser.add_argument("--code", type=str, help="Country code used for folder path (for single country analysis)")
    parser.add_argument("--all-countries", action="store_true", help="Process all countries to create comprehensive matrix")
    parser.add_argument("--vpn", type=str, default=None, help="Optional VPN country code to filter")
    parser.add_argument('--show-summary', action='store_true', help='Show additional summary plots')
    parser.add_argument("--save", action="store_true", help="Save the figures")
    parser.add_argument("--include-no-cdn", action="store_true", help="Include non-CDN providers in analysis (default: only CDN providers)")
    parser.add_argument("--min-threshold", type=float, default=3.0, help="Minimum percentage threshold for keeping providers separate (default: 3.0)")
    parser.add_argument("--class", type=int, choices=[1, 2, 3, 4], help="Filter analysis by website class (1=Critical Services, 2=News, 3=General Digital Services, 4=Entertainment)")
    return parser.parse_args()

# ===================
# CDN Matrix Functions
# ===================
def build_cdn_provider_matrix(aggregated_data, country_labels, include_no_cdn=False, min_threshold_percent=3.0):
    """
    Build a CDN provider dependency matrix from aggregated country-provider data.
    
    Args:
        aggregated_data: dict where keys are source countries and values are 
                        dicts of {provider: {"cdn": count, "no_cdn": count}}
        country_labels: list of countries that have folders in results/
        include_no_cdn: whether to include non-CDN providers in analysis
        min_threshold_percent: minimum percentage threshold for keeping providers separate
    
    Returns:
        matrix: n x m matrix of dependencies (countries x providers)
        provider_labels: list of provider names (including "Others" if applicable)
        provider_types: dict mapping provider to primary type ("cdn" or "no_cdn")
    """
    # Get all unique providers across all countries
    all_providers = set()
    for country_data in aggregated_data.values():
        all_providers.update(country_data.keys())
    
    n_countries = len(country_labels)
    
    # Create matrices for CDN and non-CDN counts
    provider_data = {}  # provider -> {"cdn": array, "no_cdn": array, "total": array}
    
    for provider in all_providers:
        provider_data[provider] = {
            "cdn": np.zeros(n_countries),
            "no_cdn": np.zeros(n_countries),
            "total": np.zeros(n_countries)
        }
    
    # Fill the provider data
    for country_idx, country in enumerate(country_labels):
        if country in aggregated_data:
            country_data = aggregated_data[country]
            
            for provider, counts in country_data.items():
                cdn_count = counts.get("cdn", 0)
                no_cdn_count = counts.get("no_cdn", 0)
                
                provider_data[provider]["cdn"][country_idx] = cdn_count
                provider_data[provider]["no_cdn"][country_idx] = no_cdn_count
                provider_data[provider]["total"][country_idx] = cdn_count + no_cdn_count
    
    # Determine which data to use based on include_no_cdn flag
    final_provider_data = {}
    provider_types = {}
    
    for provider, data in provider_data.items():
        if include_no_cdn:
            # Use total counts (CDN + non-CDN)
            final_provider_data[provider] = data["total"]
            # Determine primary type
            total_cdn = np.sum(data["cdn"])
            total_no_cdn = np.sum(data["no_cdn"])
            provider_types[provider] = "cdn" if total_cdn >= total_no_cdn else "no_cdn"
        else:
            # Use only CDN counts, skip providers with no CDN usage
            if np.sum(data["cdn"]) > 0:
                final_provider_data[provider] = data["cdn"]
                provider_types[provider] = "cdn"
    
    # Calculate normalized percentages for each provider to determine which to aggregate
    providers_to_keep = []
    providers_to_aggregate = []
    
    for provider, counts in final_provider_data.items():
        # Calculate row sums for normalization
        total_counts = np.sum(counts)
        if total_counts == 0:
            continue
            
        # Normalize to percentages for each country
        country_totals = np.zeros(n_countries)
        for country_idx, country in enumerate(country_labels):
            if country in aggregated_data:
                country_total = 0
                for p, p_counts in aggregated_data[country].items():
                    if include_no_cdn:
                        country_total += p_counts.get("cdn", 0) + p_counts.get("no_cdn", 0)
                    else:
                        country_total += p_counts.get("cdn", 0)
                country_totals[country_idx] = country_total
        
        # Avoid division by zero
        country_totals[country_totals == 0] = 1
        
        # Calculate percentages for this provider
        percentages = (counts / country_totals) * 100
        max_percentage = np.max(percentages)
        
        # Check if any country has >= threshold% dependency on this provider
        if max_percentage >= min_threshold_percent:
            providers_to_keep.append(provider)
        else:
            providers_to_aggregate.append(provider)
    
    # Build the final matrix
    kept_providers = sorted(providers_to_keep)
    final_provider_labels = kept_providers.copy()
    
    # Calculate column sums for ordering (before normalization)
    provider_sums = []
    for provider in kept_providers:
        total_sum = np.sum(final_provider_data[provider])
        provider_sums.append((provider, total_sum))
    
    # Sort providers by total sum (descending)
    provider_sums.sort(key=lambda x: x[1], reverse=True)
    ordered_providers = [p[0] for p in provider_sums]
    
    # Create the matrix with ordered providers
    has_aggregated = len(providers_to_aggregate) > 0
    n_providers = len(ordered_providers) + (1 if has_aggregated else 0)
    matrix = np.zeros((n_countries, n_providers))
    
    # Fill matrix with kept providers (in order)
    final_provider_labels = ordered_providers.copy()
    for provider_idx, provider in enumerate(ordered_providers):
        matrix[:, provider_idx] = final_provider_data[provider]
    
    # Add aggregated column if needed
    if has_aggregated:
        aggregated_column = np.zeros(n_countries)
        for provider in providers_to_aggregate:
            aggregated_column += final_provider_data[provider]
        
        matrix[:, -1] = aggregated_column
        final_provider_labels.append("Others")
        provider_types["Others"] = "aggregated"
    
    print(f"Kept {len(ordered_providers)} major providers, aggregated {len(providers_to_aggregate)} minor providers")
    
    return matrix, final_provider_labels, provider_types

def create_cdn_provider_heatmap(matrix, country_labels, provider_labels, provider_types,
                              title="Country to CDN Provider Dependency", save_path=None):
    """
    Create a CDN provider dependency heatmap visualization.
    
    Args:
        matrix: n x m matrix of dependencies (countries x providers)
        country_labels: list of country names/codes
        provider_labels: list of provider names (including "Others" if applicable)
        provider_types: dict mapping provider to type ("cdn", "no_cdn", or "aggregated")
        title: plot title
        save_path: path to save the figure (optional)
    """
    
    # Normalize the matrix
    normalized_matrix = normalize_matrix(matrix)
    
    # Set up the plot
    figsize = (max(14, len(provider_labels) * 0.8), min(10, len(country_labels) * 0.6))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create a mask for zero values to make them transparent
    mask_zeros = normalized_matrix == 0
    
    # Create the heatmap
    mapped_labels = [str.capitalize(label.lower()[0]) + label.lower()[1: 10] + ("." if len(label) > 10 else '') for label in provider_labels]
    sns.heatmap(
        normalized_matrix,
        xticklabels=mapped_labels,
        yticklabels=country_labels,
        annot=True,
        fmt='.1f',
        annot_kws={'size': 12},  # Increase cell font size
        cmap='viridis',  # Different colormap for provider analysis
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'CDN Provider Concentration (%)'},
        square=False,
        linewidths=1,
        linecolor='white',
        #mask=mask_zeros,  # This makes zero values transparent
        ax=ax
    )
    
    # Increase colorbar label font size
    cbar = ax.collections[0].colorbar
    cbar.set_label('CDN Provider Concentration (%)', fontsize=15)
    
    # Style the plot
    ax.tick_params(axis='x', which='major', labelsize=14, top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.tick_params(axis='y', which='major', labelsize=14)
    
    # Rotate the x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='left')
    
    # Color-code provider labels based on type
    x_labels = ax.get_xticklabels()
    for i, label in enumerate(x_labels):
        provider = provider_labels[i]
        provider_type = provider_types.get(provider, "unknown")
        if provider_type == "cdn":
            label.set_color('blue')
        elif provider_type == "aggregated":
            label.set_color('purple')
        else:
            label.set_color('red')
    
    # Set labels
    ax.set_xlabel('CDN Providers', fontsize=15)
    ax.set_ylabel('Countries', fontsize=15)
    
    # Add title
    #ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add legend for provider types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='CDN Providers')
    ]
    
    # Add non-CDN and aggregated to legend if they exist
    if any(provider_types.get(p) == "no_cdn" for p in provider_labels):
        legend_elements.append(Patch(facecolor='red', label='Non-CDN Providers'))
    
    if any(provider_types.get(p) == "aggregated" for p in provider_labels):
        legend_elements.append(Patch(facecolor='purple', label='Aggregated (Others)'))
    
    #ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CDN Provider heatmap saved to: {save_path}")
    
    # Show the plot
    plt.show()
    
    return normalized_matrix

def create_average_dependency_barplot(matrix, provider_labels, provider_types, 
                                    title="Average CDN Provider Dependency", save_path=None):
    """
    Create a bar plot showing average dependency percentage for each CDN provider.
    
    Args:
        matrix: n x m matrix of dependencies (countries x providers), with absolute counts
        normalized_matrix: n x m matrix of dependencies (countries x providers), with percentages
        provider_labels: list of provider names
        provider_types: dict mapping provider to type
        title: plot title
        save_path: path to save the figure (optional)
    """
    
    # Calculate average percentage for each provider (column-wise mean)
    total_counts = np.sum(matrix, axis=0)
    total_global_counts = np.sum(total_counts)

    avg_percentages = total_counts / total_global_counts * 100
    
    # Filter out "Others" or "Unknown" columns if they exist
    filtered_data = []
    filtered_labels = []
    filtered_types = []
    filtered_totals = []

    others_data = []
    others_label = []
    others_type = []
    others_total = []

    for i, (label, avg_pct, total_count) in enumerate(zip(provider_labels, avg_percentages, total_counts)):
        mapped_label = str.capitalize(label.lower()[0]) + label.lower()[1: 10] + ("." if len(label) > 10 else '')
        if label.lower() == 'others':
            others_data.append(avg_pct)
            others_label.append(mapped_label)
            others_type.append(provider_types.get(label, 'unknown'))
            others_total.append(total_count)

        elif label.lower() not in ['unknown']:
            filtered_data.append(avg_pct)
            filtered_labels.append(mapped_label)
            filtered_types.append(provider_types.get(label, 'unknown'))
            filtered_totals.append(total_count)
    
    # Sort by average percentage (descending)
    sorted_data = sorted(zip(filtered_data, filtered_labels, filtered_types, filtered_totals), reverse=True)
    sorted_percentages, sorted_labels, sorted_types, sorted_totals = zip(*sorted_data)

    sorted_percentages = sorted_percentages + tuple(others_data)
    sorted_labels = sorted_labels + tuple(others_label)
    sorted_types = sorted_types + tuple(others_type)
    sorted_totals = sorted_totals + tuple(others_total)

    # Create the plot
    plt.figure(figsize=(max(12, len(sorted_labels) * 0.8), 5))
    
    # Create color map based on provider types
    colors = []
    for ptype in sorted_types:
        if ptype == "cdn":
            colors.append('#1f77b4')  # blue
        elif ptype == "no_cdn":
            colors.append('#d62728')  # red
        else:
            colors.append('#ff7f0e')  # orange
    
    bars = plt.bar(range(len(sorted_labels)), sorted_totals, color=colors)
    
    # Customize the plot
    #plt.xlabel('CDN Providers', fontsize=18)
    plt.ylabel('Amount of Websites', fontsize=18)
    #plt.title(title, fontsize=16, fontweight='bold')
    plt.ylim(0, max(sorted_totals) * 1.1)

    # Set x-axis labels
    plt.xticks(range(len(sorted_labels)), sorted_labels, rotation=45, ha='right', fontsize=15)
    plt.yticks(fontsize=15)

    # Add percentage labels on top of bars
    for bar, pct in zip(bars, sorted_percentages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=13)
    
    # Add legend if we have different provider types
    unique_types = list(set(sorted_types))
    if len(unique_types) > 1:
        from matplotlib.patches import Patch
        legend_elements = [] if len(others_data) == 0 else [Patch(facecolor='#ff7f0e', label='Grouped CDN Providers')]
        if 'cdn' in unique_types:
            legend_elements.append(Patch(facecolor='#1f77b4', label='CDN Providers'))
        if 'no_cdn' in unique_types:
            legend_elements.append(Patch(facecolor='#d62728', label='Non-CDN Providers'))
        plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Average dependency bar plot saved to: {save_path}")
    
    plt.show()
    
    return sorted_percentages, sorted_labels

# ===================
# Main Function
# ===================
def main():
    """
    Main function that processes CDN provider data and creates dependency heatmap
    """
    args = parse_arguments()
    
    # Load classified websites data if class filtering is requested
    domain_to_class = None
    class_filter = None
    
    if getattr(args, 'class', None) is not None:
        print(f"Loading classified websites data for class filtering...")
        try:
            domain_to_class = load_classified_websites("classified_websites.json")
            class_mapping = get_class_mapping()
            # Find the class name corresponding to the numerical class
            class_names = {v: k for k, v in class_mapping.items()}
            class_filter = class_names.get(getattr(args, 'class'))
            
            if class_filter:
                print(f"Filtering to class {getattr(args, 'class')}: {class_filter}")
                print(f"Loaded {len(domain_to_class)} classified domains")
            else:
                print(f"Warning: Invalid class number {getattr(args, 'class')}")
                return
                
        except FileNotFoundError:
            print("Error: classified_websites.json not found. Please ensure the file exists in the current directory.")
            return
        except Exception as e:
            print(f"Error loading classified websites: {e}")
            return
    
    # Validate arguments
    if args.all_countries:
        if args.country or args.code:
            print("Warning: --country and --code arguments ignored when using --all-countries")
    else:
        if not args.country or not args.code:
            print("Error: Either use --all-countries or provide both --country and --code")
            return

    # Determine which countries to process
    if args.all_countries:
        country_codes = get_all_country_codes("results")
        if not country_codes:
            print("No countries found in results directory")
            return
        print(f"Found {len(country_codes)} countries: {', '.join(country_codes)}")
    else:
        country_codes = [args.code]

    # Process all countries to build comprehensive CDN provider data
    all_provider_data = defaultdict(lambda: defaultdict(lambda: {"cdn": 0, "no_cdn": 0}))
    total_unknown = 0
    total_experiments = 0

    for country_code in country_codes:
        print(f"\nProcessing country: {country_code}")
        base_path = Path(f"results/{country_code}/locality")

        # Find all experiment files for this country
        file_types = ["locedge.json", "cdn.json", "provider.json"]
        experiment_paths = find_experiment_files(base_path, file_types, args.vpn)

        print(f"  Found {len(experiment_paths)} experiments with CDN data")
        
        # Process all experiments for this country
        for locedge_path, cdn_path, provider_path in experiment_paths:
            with open(locedge_path) as f:
                locedge_data = json.load(f)
            with open(cdn_path) as f:
                cdn_data = json.load(f)
            with open(provider_path) as f:
                provider_data = json.load(f)

            # Process this experiment to get provider counts
            provider_counts = defaultdict(lambda: {"cdn": 0, "no_cdn": 0})
            unknown_count = process_experiment_cdn(cdn_data, locedge_data, provider_data, provider_counts, class_filter, domain_to_class)
            total_unknown += unknown_count
            total_experiments += 1
            
            # Add to aggregated data (source country is the current country_code)
            for provider, counts in provider_counts.items():
                all_provider_data[country_code][provider]["cdn"] += counts["cdn"]
                all_provider_data[country_code][provider]["no_cdn"] += counts["no_cdn"]

        # Print summary for this country
        country_total = sum(
            sum(provider_counts.values()) 
            for provider_counts in all_provider_data[country_code].values()
        )
        print(f"  Total provider dependencies: {country_total}")

    if total_experiments == 0:
        print("No experiment files found")
        return

    # Use ONLY the countries we actually studied (have folders) as labels
    if args.all_countries:
        country_codes_sorted = sort_countries_by_continent(country_codes)  # Sort by continent
        country_labels = convert_codes_to_names(country_codes_sorted)  # Convert to full names
    else:
        country_codes_sorted = [args.code]  # Single country analysis
        country_labels = convert_codes_to_names(country_codes_sorted)
    
    print(f"\nMatrix will include these studied countries (by continent): {', '.join(country_labels)}")
    
    # Show continent groupings for better understanding
    if args.all_countries:
        continent_mapping = get_continent_mapping()
        continent_groups = {}
        for code, name in zip(country_codes_sorted, country_labels):
            continent = continent_mapping.get(code, 'Other')
            if continent not in continent_groups:
                continent_groups[continent] = []
            continent_groups[continent].append(name)
        
        print("\nCountries grouped by continent:")
        continent_order = ['North America', 'Central America', 'South America', 'Europe', 'Asia', 'Africa', 'Oceania', 'Other']
        for continent in continent_order:
            if continent in continent_groups:
                print(f"  {continent}: {', '.join(continent_groups[continent])}")
    
    # Build the CDN provider matrix
    matrix, provider_labels, provider_types = build_cdn_provider_matrix(
        all_provider_data, 
        country_codes_sorted,
        include_no_cdn=args.include_no_cdn,
        min_threshold_percent=args.min_threshold
    )
    
    print(f"Found {len(provider_labels)} unique providers: {', '.join(provider_labels[:10])}{'...' if len(provider_labels) > 10 else ''}")
    
    # Create title
    if args.all_countries:
        title = "Global CDN Provider Dependency Analysis"
        if args.vpn:
            title += f" (VPN: {args.vpn})"
    else:
        title = f"CDN Provider Dependency Analysis - {args.country}"
        if args.vpn:
            title += f" (VPN: {args.vpn})"
    
    # Add class information to title
    if class_filter:
        class_names = {
            "1. Critical & Social Services": "Critical Services",
            "4. Media & News": "News", 
            "3. Commerce, Retail & Industry": "General Digital Services",
            "5. Entertainment & Social Media": "Entertainment"
        }
        short_name = class_names.get(class_filter, class_filter)
        title += f" - Class {getattr(args, 'class')}: {short_name}"
    
    # Generate output path if saving
    heatmap_path = None
    if args.save:
        vpn_suffix = f"_vpn_{args.vpn}" if args.vpn else ""
        class_suffix = f"_class_{getattr(args, 'class')}" if class_filter else ""
        
        if args.all_countries:
            output_dir = Path("results")
            output_dir.mkdir(parents=True, exist_ok=True)
            heatmap_path = output_dir / f"global_cdn_provider_heatmap{vpn_suffix}{class_suffix}.png"
        else:
            output_dir = Path(f"results/{args.code}/results/locality")
            output_dir.mkdir(parents=True, exist_ok=True)
            heatmap_path = output_dir / f"cdn_provider_heatmap{vpn_suffix}{class_suffix}.png"
    
    # Create CDN provider heatmap
    print("\n=== Creating CDN Provider Heatmap ===")
    normalized_matrix = create_cdn_provider_heatmap(
        matrix, 
        country_labels,
        provider_labels,
        provider_types,
        title=title,
        save_path=heatmap_path
    )
    
    # Create average dependency bar plot
    print("\n=== Creating Average Dependency Bar Plot ===")
    bar_title = "Average CDN Provider Dependency Across Countries"
    if args.vpn:
        bar_title += f" (VPN: {args.vpn})"
    if class_filter:
        class_names = {
            "1. Critical & Social Services": "Critical Services",
            "4. Media & News": "News", 
            "3. Commerce, Retail & Industry": "General Digital Services",
            "5. Entertainment & Social Media": "Entertainment"
        }
        short_name = class_names.get(class_filter, class_filter)
        bar_title += f" - Class {getattr(args, 'class')}: {short_name}"
    
    bar_path = None
    if args.save:
        if args.all_countries:
            bar_path = output_dir / f"global_cdn_average_dependency{vpn_suffix}{class_suffix}.png"
        else:
            bar_path = output_dir / f"cdn_average_dependency{vpn_suffix}{class_suffix}.png"
    
    avg_percentages, sorted_labels = create_average_dependency_barplot(
        matrix,
        provider_labels,
        provider_types,
        title=bar_title,
        save_path=bar_path
    )
    
    # Print comprehensive statistics
    print("\n=== CDN Provider Statistics ===")
    print(f"Matrix shape: {normalized_matrix.shape}")
    print(f"Countries analyzed: {len(country_labels)}")
    print(f"Providers found: {len(provider_labels)}")
    print(f"Include non-CDN providers: {args.include_no_cdn}")
    print(f"Minimum threshold for separate display: {args.min_threshold}%")
    print(f"Total experiment files processed: {total_experiments}")
    print(f"Total unknown providers: {total_unknown}")
    
    # Show top providers by usage
    provider_totals = np.sum(matrix, axis=0)
    top_providers_with_totals = list(zip(provider_labels, provider_totals))
    # Don't sort again since they're already ordered, just show top 10 or all if fewer
    top_providers = top_providers_with_totals[:min(10, len(top_providers_with_totals))]
    
    print(f"\nTop Providers by Total Usage:")
    for i, (provider, total) in enumerate(top_providers, 1):
        provider_type = provider_types.get(provider, "unknown")
        if provider == "Others":
            print(f"  {i}. {provider} (aggregated): {total:.0f} total uses")
        else:
            print(f"  {i}. {provider} ({provider_type}): {total:.0f} total uses")


if __name__ == "__main__":
    main()