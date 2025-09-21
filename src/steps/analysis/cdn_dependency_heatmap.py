import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from pathlib import Path
from collections import defaultdict

from src.steps.analysis.helpers import (
    get_all_country_codes, convert_codes_to_names, sort_countries_by_continent,
    get_continent_mapping, process_experiment_cdn, normalize_matrix, find_experiment_files
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
    figsize = (max(16, len(provider_labels) * 0.8), max(12, len(country_labels) * 0.6))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create a mask for zero values to make them transparent
    mask_zeros = normalized_matrix == 0
    
    # Create the heatmap
    sns.heatmap(
        normalized_matrix,
        xticklabels=provider_labels,
        yticklabels=country_labels,
        annot=True,
        fmt='.1f',
        cmap='plasma',  # Different colormap for provider analysis
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Provider Usage Percentage (%)'},
        square=False,
        linewidths=0.5,
        linecolor='white',
        mask=mask_zeros,  # This makes zero values transparent
        ax=ax
    )
    
    # Style the plot
    ax.tick_params(axis='x', which='major', labelsize=9, top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.tick_params(axis='y', which='major', labelsize=9)
    
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
    ax.set_xlabel('CDN/Content Providers')
    ax.set_ylabel('Countries')
    
    # Add title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
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
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CDN Provider heatmap saved to: {save_path}")
    
    # Show the plot
    plt.show()
    
    return normalized_matrix

# ===================
# Main Function
# ===================
def main():
    """
    Main function that processes CDN provider data and creates dependency heatmap
    """
    args = parse_arguments()
    
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
            unknown_count = process_experiment_cdn(cdn_data, locedge_data, provider_data, provider_counts)
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
    
    # Generate output path if saving
    heatmap_path = None
    if args.save:
        if args.all_countries:
            output_dir = Path("results")
            output_dir.mkdir(parents=True, exist_ok=True)
            vpn_suffix = f"_vpn_{args.vpn}" if args.vpn else ""
            heatmap_path = output_dir / f"global_cdn_provider_heatmap{vpn_suffix}.png"
        else:
            output_dir = Path(f"results/{args.code}/results/locality")
            output_dir.mkdir(parents=True, exist_ok=True)
            vpn_suffix = f"_vpn_{args.vpn}" if args.vpn else ""
            heatmap_path = output_dir / f"cdn_provider_heatmap{vpn_suffix}.png"
    
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