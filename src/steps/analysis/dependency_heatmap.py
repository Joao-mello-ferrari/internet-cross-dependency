import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openchord_modified as ocd
import argparse
import json
import glob
from pathlib import Path
from collections import defaultdict

# ===================
# Argument Parser
# ===================
def parse_arguments():
    parser = argparse.ArgumentParser(description='Create dependency matrix heatmap and chord visualizations')
    parser.add_argument("--country", type=str, help="Country label (for single country analysis)")
    parser.add_argument("--code", type=str, help="Country code used for folder path (for single country analysis)")
    parser.add_argument("--all-countries", action="store_true", help="Process all countries to create comprehensive matrix")
    parser.add_argument("--vpn", type=str, default=None, help="Optional VPN country code to filter")
    parser.add_argument('--show-summary', action='store_true', help='Show additional summary plots')
    parser.add_argument("--save", action="store_true", help="Save the figures")
    parser.add_argument("--heatmap-only", action="store_true", help="Generate only heatmap (skip chord diagram)")
    parser.add_argument("--chord-only", action="store_true", help="Generate only chord diagram (skip heatmap)")
    parser.add_argument("--cdn-providers", action="store_true", help="Analyze CDN provider dependencies instead of country dependencies")
    parser.add_argument("--include-no-cdn", action="store_true", help="Include non-CDN providers in analysis (default: only CDN providers)")
    return parser.parse_args()


# ===================
# Helper Functions
# ===================
def get_all_country_codes(results_base_path):
    """
    Get all country codes from the results directory.
    
    Args:
        results_base_path: Path to the results directory
    
    Returns:
        list of country codes (folder names that contain locality data)
    """
    results_path = Path(results_base_path)
    country_codes = []
    
    for item in results_path.iterdir():
        if item.is_dir() and not item.name.endswith('.json'):
            # Check if this country has locality data
            locality_path = item / "locality"
            if locality_path.exists():
                country_codes.append(item.name)
    
    return sorted(country_codes)

def get_country_name_mapping():
    """
    Get mapping from country codes to full country names.
    
    Returns:
        dict mapping country codes to country names
    """
    country_mapping = {
        'ar': 'Argentina',
        'au': 'Australia', 
        'br': 'Brazil',
        'ca': 'Canada',
        'co': 'Colombia',
        'cr': 'Costa Rica',
        'de': 'Germany',
        'do': 'Dominican Republic',
        'eg': 'Egypt',
        'es': 'Spain',
        'fr': 'France',
        'gb': 'United Kingdom',
        'gt': 'Guatemala',
        'id': 'Indonesia',
        'in': 'India',
        'it': 'Italy',
        'jp': 'Japan',
        'mx': 'Mexico',
        'ng': 'Nigeria',
        'nz': 'New Zealand',
        'pg': 'Papua New Guinea',
        'us': 'United States',
        'za': 'South Africa'
    }
    return country_mapping

def convert_codes_to_names(country_codes):
    """
    Convert country codes to full names where possible.
    
    Args:
        country_codes: list of country codes
        
    Returns:
        list of country names (or codes if name not found)
    """
    mapping = get_country_name_mapping()
    return [mapping.get(code, code.upper()) for code in country_codes]

def get_continent_mapping():
    """
    Get mapping from country codes to continents.
    
    Returns:
        dict mapping country codes to continent names
    """
    continent_mapping = {
        # North America
        'us': 'North America',
        'ca': 'North America', 
        'mx': 'North America',
        
        # Central America (Middle America)
        'gt': 'Central America',
        'cr': 'Central America',
        'do': 'Central America',
        
        # South America
        'ar': 'South America',
        'br': 'South America',
        'co': 'South America',
        
        # Europe
        'de': 'Europe',
        'es': 'Europe',
        'fr': 'Europe',
        'gb': 'Europe',
        'it': 'Europe',
        
        # Asia
        'id': 'Asia',
        'in': 'Asia',
        'jp': 'Asia',
        
        # Africa
        'eg': 'Africa',
        'ng': 'Africa',
        'za': 'Africa',
        
        # Oceania
        'au': 'Oceania',
        'nz': 'Oceania',
        'pg': 'Oceania'
    }
    return continent_mapping

def sort_countries_by_continent(country_codes):
    """
    Sort country codes by continent, then alphabetically within each continent.
    
    Args:
        country_codes: list of country codes
        
    Returns:
        list of country codes sorted by continent
    """
    continent_mapping = get_continent_mapping()
    
    # Define continent order
    continent_order = ['North America', 'Central America', 'South America', 'Europe', 'Asia', 'Africa', 'Oceania']
    
    # Group countries by continent
    countries_by_continent = {}
    for continent in continent_order:
        countries_by_continent[continent] = []
    
    # Add an "Other" category for unmapped countries
    countries_by_continent['Other'] = []
    
    for code in country_codes:
        continent = continent_mapping.get(code, 'Other')
        countries_by_continent[continent].append(code)
    
    # Sort countries within each continent alphabetically
    for continent in countries_by_continent:
        countries_by_continent[continent].sort()
    
    # Combine all continents in order
    sorted_codes = []
    for continent in continent_order + ['Other']:
        sorted_codes.extend(countries_by_continent[continent])
    
    return sorted_codes

def process_experiment(location_data, locedge_data, dependency_counts):
    """
    Process a single experiment to count dependencies between countries.
    
    Args:
        location_data: dict mapping domains to their locations
        locedge_data: dict with additional location/edge data
        dependency_counts: nested dict to accumulate dependency counts
    
    Returns:
        unknown_count: number of domains with unknown location
    """
    unknown_count = 0
    for domain, location in location_data.items():
        content_location = locedge_data.get(domain, {}).get("contentLocality")

        if location is None and content_location is None:
            unknown_count += 1
            continue

        # Use content_location if available, otherwise use location
        final_location = content_location if content_location else location
        
        # Clean up location (remove anycast suffix if present)
        if "- anycast" in final_location:
            unknown_count += 1
            continue
            # final_location = final_location.replace(" - anycast", "")
        
        # Count this dependency
        if final_location not in dependency_counts:
            dependency_counts[final_location] = 0
        dependency_counts[final_location] += 1

    return unknown_count

def process_experiment_cdn(cdn_data, locedge_data, provider_data, provider_counts):
    """
    Process a single experiment to count dependencies on CDN providers.
    
    Args:
        cdn_data: dict mapping domains to their CDN providers
        locedge_data: dict with additional location/edge data
        provider_data: dict mapping domains to provider info from whois
        provider_counts: nested dict to accumulate provider counts
    
    Returns:
        unknown_count: number of domains with unknown provider
    """
    unknown_count = 0
    for domain, cdn_string in cdn_data.items():
        by_whois_provider = provider_data.get(domain)
        cdn_providers = set(cdn_string.replace("'", "").split(", ")) if cdn_string else []
        no_cdn_providers = set(locedge_data.get(domain, {}).get("provider", []) + [by_whois_provider] if by_whois_provider is not None else [])

        if len(cdn_providers) == 0 and len(no_cdn_providers) == 0:
            unknown_count += 1
            continue

        if cdn_providers:
            for provider in map(str.lower, cdn_providers):
                if provider == "aws (not cdn)":
                    continue
                provider_counts[provider]["cdn"] += 1
        else:
            for provider in map(str.lower, no_cdn_providers):
                if provider == "aws (not cdn)":
                    continue
                provider_counts[provider]["no_cdn"] += 1

    return unknown_count

def build_dependency_matrix(aggregated_data, country_labels):
    """
    Build a dependency matrix from aggregated country dependency data.
    
    Args:
        aggregated_data: dict where keys are source countries and values are 
                        dicts of {target_country: count}
        country_labels: list of ONLY the countries that have folders in results/ 
                       (i.e., countries we actually studied)
    
    Returns:
        matrix: n x n matrix of dependencies (only for studied countries)
        unmapped_dependencies: list of dependencies to countries NOT in country_labels
    """
    n_countries = len(country_labels)
    matrix = np.zeros((n_countries, n_countries))
    unmapped_dependencies = np.zeros(n_countries)
    
    # Create mapping from country name to index (only for studied countries)
    country_to_idx = {country: i for i, country in enumerate(country_labels)}
    
    for source_idx, source_country in enumerate(country_labels):
        if source_country in aggregated_data:
            source_data = aggregated_data[source_country]
            
            for target_country, count in source_data.items():
                if target_country in country_to_idx:
                    # Dependency on a country we studied (has folder in results/)
                    target_idx = country_to_idx[target_country]
                    matrix[source_idx, target_idx] = count
                else:
                    # Dependency on a country we did NOT study (unmapped)
                    unmapped_dependencies[source_idx] += count
    
    return matrix, unmapped_dependencies

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

def normalize_provider_matrix(matrix):
    """
    Normalize the provider matrix so each row sums to 100%.
    
    Args:
        matrix: n x m matrix of dependencies (countries x providers)
    
    Returns:
        normalized_matrix: n x m matrix with percentages
    """
    matrix = np.array(matrix, dtype=float)
    
    # Calculate row sums
    row_sums = np.sum(matrix, axis=1)
    
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    
    # Normalize each row to sum to 100%
    normalized_matrix = (matrix / row_sums[:, np.newaxis]) * 100
    
    return normalized_matrix

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
    normalized_matrix = normalize_provider_matrix(matrix)
    
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
def normalize_matrix(matrix, unmapped_dependencies):
    """
    Normalize the matrix so each row sums to 100%.
    
    Args:
        matrix: n x n matrix of dependencies between countries
        unmapped_dependencies: list of size n with unmapped dependencies
    
    Returns:
        normalized_matrix: n x (n+1) matrix with percentages
    """
    matrix = np.array(matrix, dtype=float)
    unmapped_dependencies = np.array(unmapped_dependencies, dtype=float)
    
    # Combine matrix with unmapped dependencies as last column
    extended_matrix = np.column_stack([matrix, unmapped_dependencies])
    
    # Calculate row sums
    row_sums = np.sum(extended_matrix, axis=1)
    
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    
    # Normalize each row to sum to 100%
    normalized_matrix = (extended_matrix / row_sums[:, np.newaxis]) * 100
    
    return normalized_matrix

def create_dependency_heatmap(matrix, unmapped_dependencies, country_labels, 
                            title="Country Dependency Matrix", save_path=None):
    """
    Create a dependency heatmap visualization.
    
    Args:
        matrix: n x n matrix of dependencies between countries
        unmapped_dependencies: list of size n with unmapped dependencies  
        country_labels: list of country names/codes
        title: plot title
        save_path: path to save the figure (optional)
    """
    
    # Normalize the matrix
    normalized_matrix = normalize_matrix(matrix, unmapped_dependencies)
    
    # Set up the plot with subplots for separation
    figsize = (16, 12) if len(country_labels) > 10 else (12, 10)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 1.1, figsize[1]), 
                                   gridspec_kw={'width_ratios': [len(country_labels), 1], 'wspace': 0.02})
    
    # Create the main country-to-country heatmap
    country_matrix = normalized_matrix[:, :-1]  # All columns except last (unmapped)
    
    # Create a mask for zero values to make them transparent
    mask_zeros = country_matrix == 0
    
    sns.heatmap(
        country_matrix,
        xticklabels=country_labels,
        yticklabels=country_labels,
        annot=True,
        fmt='.1f',
        cmap='viridis',
        vmin=0,
        vmax=100,
        cbar=False,  # We'll add a shared colorbar later
        square=False,
        linewidths=0.5,
        linecolor='white',
        mask=mask_zeros,  # This makes zero values transparent
        ax=ax1
    )
    
    # Create the unmapped dependencies heatmap (single column)
    unmapped_matrix = normalized_matrix[:, -1:] # Last column only
    
    # Create a mask for zero values in unmapped column
    mask_unmapped_zeros = unmapped_matrix == 0
    
    sns.heatmap(
        unmapped_matrix,
        xticklabels=["Unmapped"],
        yticklabels=False,  # Don't repeat country labels
        annot=True,
        fmt='.1f',
        cmap='viridis',
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Dependency Percentage (%)'},
        square=False,
        linewidths=0.5,
        linecolor='white',
        mask=mask_unmapped_zeros,  # This makes zero values transparent
        ax=ax2
    )
    
    # Style the plots
    ax1.tick_params(axis='x', which='major', labelsize=9, top=True, bottom=False, labeltop=True, labelbottom=False)
    ax1.tick_params(axis='y', which='major', labelsize=9)
    ax2.tick_params(axis='x', which='major', labelsize=9, top=True, bottom=False, labeltop=True, labelbottom=False)
    ax2.tick_params(axis='y', which='major', labelsize=9, left=False, labelleft=False)
    
    # Rotate the x-axis labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='left')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='left')
    
    # Set labels
    ax1.set_xlabel('')
    ax1.set_ylabel('Dependent Country')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    
    # Add a main title spanning both subplots
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    # Show the plot
    plt.show()
    
    return normalized_matrix


def create_dependency_chord(matrix, country_labels, save_path=None):
    """
    Create a dependency chord diagram visualization.
    
    Args:
        matrix: n x n matrix of dependencies between countries
        country_labels: list of country names/codes
        save_path: path to save the figure (optional)
    """
    
    # Create a copy to avoid modifying the original matrix
    matrix_processed = matrix.copy()
    
    # Make matrix symmetric by adding small values to zero cells in transpose positions
    for n in range(matrix_processed.shape[0]):
        for m in range(matrix_processed.shape[1]):
            # If cell (n,m) has a value and cell (m,n) is zero, add small value to (m,n)
            if matrix_processed[n, m] != 0 and matrix_processed[m, n] == 0:
                matrix_processed[m, n] = matrix_processed[n, m] * 0.01
    
    # Calculate column sums and apply cubic root transformation for gentle scaling
    col_sums = np.sum(matrix_processed, axis=0)
    col_sums_cbrt = np.cbrt(col_sums + 1e-10)  # Add tiny value for numerical stability
    
    # Apply scaling
    matrix_scaled = matrix_processed * col_sums_cbrt[:, np.newaxis]
    
    # Create and display the chord diagram
    print(f"Creating chord diagram with {len(country_labels)} countries...")
    print(f"Original column sums: {col_sums}")
    print(f"Scaled column sums: {col_sums_cbrt}")
    
    fig = ocd.Chord(matrix_scaled, country_labels)
    fig.show()
    
    # Save if path provided
    if save_path:
        fig.save_svg(save_path)
        print(f"Chord diagram saved to: {save_path}")
    
    return matrix_scaled
    
    



def create_summary_plots(normalized_matrix, country_labels):
    """
    Create additional summary visualizations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Total dependency on others (excluding self)
    total_dependency = np.sum(normalized_matrix[:, :-1], axis=1) - np.diag(normalized_matrix[:, :-1])
    axes[0, 0].barh(country_labels, total_dependency)
    axes[0, 0].set_title('Total Dependency on Other Countries (%)')
    axes[0, 0].set_xlabel('Percentage (%)')
    
    # 2. Unmapped dependency
    unmapped_dep = normalized_matrix[:, -1]
    axes[0, 1].barh(country_labels, unmapped_dep)
    axes[0, 1].set_title('Unmapped Dependency (%)')
    axes[0, 1].set_xlabel('Percentage (%)')
    
    # 3. Self-dependency
    self_dependency = np.diag(normalized_matrix[:, :-1])
    axes[1, 0].barh(country_labels, self_dependency)
    axes[1, 0].set_title('Self Dependency (%)')
    axes[1, 0].set_xlabel('Percentage (%)')
    
    # 4. How much others depend on each country (column sums)
    dependency_caused = np.sum(normalized_matrix[:, :-1], axis=0) - np.diag(normalized_matrix[:, :-1])
    axes[1, 1].barh(country_labels, dependency_caused)
    axes[1, 1].set_title('Dependency Caused to Others (%)')
    axes[1, 1].set_xlabel('Total Percentage (%)')
    
    plt.tight_layout()
    plt.show()


# ===================
# Main Function
# ===================
def main():
    """
    Main function that processes locality data and creates dependency heatmap
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

    # Process all countries to build comprehensive dependency data
    if args.cdn_providers:
        # CDN Provider analysis
        all_provider_data = defaultdict(lambda: defaultdict(lambda: {"cdn": 0, "no_cdn": 0}))
        total_unknown = 0
        total_experiments = 0

        for country_code in country_codes:
            print(f"\nProcessing country: {country_code}")
            base_path = Path(f"results/{country_code}/locality")

            # Find all experiment files for this country
            experiment_paths = []
            for location_file in glob.glob(str(base_path / "**/location.json"), recursive=True):
                if args.vpn and args.vpn not in location_file:
                    continue

                experiment_dir = Path(location_file).parent
                locedge_file = experiment_dir / "locedge.json"
                cdn_file = experiment_dir / "cdn.json"
                provider_file = experiment_dir / "provider.json"
                
                if locedge_file.exists() and cdn_file.exists() and provider_file.exists():
                    experiment_paths.append((locedge_file, cdn_file, provider_file))

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

    else:
        # Country dependency analysis (original logic)
        all_dependency_data = defaultdict(lambda: defaultdict(int))
        total_unknown = 0
        total_experiments = 0

        for country_code in country_codes:
            print(f"\nProcessing country: {country_code}")
            base_path = Path(f"results/{country_code}/locality")

            # Find all experiment files for this country
            experiment_paths = []
            for location_file in glob.glob(str(base_path / "**/location.json"), recursive=True):
                if args.vpn and args.vpn not in location_file:
                    continue

                experiment_dir = Path(location_file).parent
                locedge_file = experiment_dir / "locedge.json"
                if locedge_file.exists():
                    experiment_paths.append((location_file, locedge_file))

            print(f"  Found {len(experiment_paths)} experiments")
            
            # Process all experiments for this country
            for location_path, locedge_path in experiment_paths:
                with open(location_path) as f:
                    location_data = json.load(f)
                with open(locedge_path) as f:
                    locedge_data = json.load(f)

                # Process this experiment to get dependency counts
                dependency_counts = defaultdict(int)
                unknown_count = process_experiment(location_data, locedge_data, dependency_counts)
                total_unknown += unknown_count
                total_experiments += 1
                
                # Add to aggregated data (source country is the current country_code)
                for target_country, count in dependency_counts.items():
                    all_dependency_data[country_code][target_country] += count

            # Print summary for this country
            country_total = sum(all_dependency_data[country_code].values())
            print(f"  Total dependencies: {country_total}")

    if total_experiments == 0:
        print("No experiment files found")
        return

    # Use ONLY the countries we actually studied (have folders) as labels
    if args.all_countries:
        country_codes_sorted = sort_countries_by_continent(country_codes)  # Sort by continent
        country_labels = convert_codes_to_names(country_codes_sorted)  # Convert to full names
        # Keep a mapping for matrix building (which still needs codes)
        code_to_name_map = dict(zip(country_codes_sorted, country_labels))
    else:
        country_codes_sorted = [args.code]  # Single country analysis
        country_labels = convert_codes_to_names(country_codes_sorted)
        code_to_name_map = dict(zip(country_codes_sorted, country_labels))
    
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
    
    # Branch based on analysis type
    if args.cdn_providers:
        # CDN Provider Analysis
        print("\n=== CDN Provider Analysis ===")
        
        # Build the CDN provider matrix
        matrix, provider_labels, provider_types = build_cdn_provider_matrix(
            all_provider_data, 
            country_codes_sorted,
            include_no_cdn=args.include_no_cdn,
            min_threshold_percent=3.0
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
        print(f"Minimum threshold for separate display: 3.0%")
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
    
    else:
        # Country Dependency Analysis (original logic)
        print("\n=== Country Dependency Analysis ===")
        
        # Build the dependency matrix (still using codes internally)
        matrix, unmapped_dependencies = build_dependency_matrix(all_dependency_data, country_codes_sorted)
        
        # Show what countries appear in data but are not studied (go to unmapped)
        all_target_countries = set()
        for source_data in all_dependency_data.values():
            all_target_countries.update(source_data.keys())
        
        unmapped_countries = all_target_countries - set(country_codes_sorted)
        if unmapped_countries:
            print(f"Countries in data but not studied (going to unmapped): {', '.join(sorted(unmapped_countries))}")
        else:
            print("All target countries are studied countries (no additional unmapped countries)")
        
        # Create title
        if args.all_countries:
            title = "Global Country Dependency Analysis"
            if args.vpn:
                title += f" (VPN: {args.vpn})"
        else:
            title = f"Country Dependency Analysis - {args.country}"
            if args.vpn:
                title += f" (VPN: {args.vpn})"
        
        # Generate output paths if saving
        heatmap_path = None
        chord_path = None
        
        if args.save:
            if args.all_countries:
                output_dir = Path("results")
                output_dir.mkdir(parents=True, exist_ok=True)
                vpn_suffix = f"_vpn_{args.vpn}" if args.vpn else ""
                heatmap_path = output_dir / f"global_dependency_heatmap{vpn_suffix}.png"
                chord_path = output_dir / f"global_dependency_chord{vpn_suffix}.svg"
            else:
                output_dir = Path(f"results/{args.code}/results/locality")
                output_dir.mkdir(parents=True, exist_ok=True)
                vpn_suffix = f"_vpn_{args.vpn}" if args.vpn else ""
                heatmap_path = output_dir / f"dependency_heatmap{vpn_suffix}.png"
                chord_path = output_dir / f"dependency_chord{vpn_suffix}.svg"
        
        # Create visualizations based on arguments
        normalized_matrix = None
        
        if not args.chord_only:
            print("\n=== Creating Dependency Heatmap ===")
            normalized_matrix = create_dependency_heatmap(
                matrix, 
                unmapped_dependencies, 
                country_labels,
                title=title + " - Heatmap",
                save_path=heatmap_path
            )
        
        if not args.heatmap_only:
            print("\n=== Creating Dependency Chord Diagram ===")
            create_dependency_chord(
                matrix, 
                country_labels,
                save_path=chord_path
            )
        
        # Create summary plots only if flag is set
        if args.show_summary and normalized_matrix is not None:
            print("\n=== Creating Summary Plots ===")
            create_summary_plots(normalized_matrix, country_labels)
        
        # Print comprehensive statistics
        if normalized_matrix is not None:
            print("\n=== Dependency Statistics ===")
            print(f"Matrix shape: {normalized_matrix.shape}")
            print(f"Countries analyzed: {len(country_labels)}")
            print(f"Source countries processed: {len(country_codes)}")
            print(f"Total experiment files processed: {total_experiments}")
            print(f"Total unknown locations: {total_unknown}")
            
            print(f"\nCountry Dependencies:")
            for i, country in enumerate(country_labels):
                if i < len(normalized_matrix):
                    self_dep = normalized_matrix[i, i] if i < normalized_matrix.shape[1] - 1 else 0
                    unmapped_dep = normalized_matrix[i, -1]
                    other_dep = 100 - self_dep - unmapped_dep
                    print(f"{country.upper()}: Self={self_dep:.1f}%, Others={other_dep:.1f}%, Unmapped={unmapped_dep:.1f}%")


if __name__ == "__main__":
    main()