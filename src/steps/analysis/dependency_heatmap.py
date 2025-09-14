import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import glob
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from collections import defaultdict

# ===================
# Argument Parser
# ===================
def parse_arguments():
    parser = argparse.ArgumentParser(description='Create dependency matrix heatmap visualization')
    parser.add_argument("--country", type=str, help="Country label (for single country analysis)")
    parser.add_argument("--code", type=str, help="Country code used for folder path (for single country analysis)")
    parser.add_argument("--all-countries", action="store_true", help="Process all countries to create comprehensive matrix")
    parser.add_argument("--vpn", type=str, default=None, help="Optional VPN country code to filter")
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--title', type=str, default='Country Dependency Matrix', help='Plot title')
    parser.add_argument('--show-summary', action='store_true', help='Show additional summary plots')
    parser.add_argument('--colormap', type=str, default='viridis', help='Matplotlib colormap name')
    parser.add_argument('--show-values', action='store_true', help='Show percentage values in cells')
    parser.add_argument("--save", action="store_true", help="Save the figure")
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
        'mx': 'Central America',
        
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
                            title="Country Dependency Matrix", figsize=(12, 10),
                            save_path=None, colormap='viridis', show_values=True):
    """
    Create a dependency heatmap visualization.
    
    Args:
        matrix: n x n matrix of dependencies between countries
        unmapped_dependencies: list of size n with unmapped dependencies  
        country_labels: list of country names/codes
        title: plot title
        figsize: figure size tuple
        save_path: path to save the figure (optional)
        colormap: matplotlib colormap name
        show_values: whether to show percentage values in cells
    """
    
    # Normalize the matrix
    normalized_matrix = normalize_matrix(matrix, unmapped_dependencies)
    
    # Create extended labels (countries + "Unmapped")
    extended_labels = country_labels + ["Unmapped"]
    
    # Set up the plot with subplots for separation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 1.1, figsize[1]), 
                                   gridspec_kw={'width_ratios': [len(country_labels), 1], 'wspace': 0.02})
    
    # Create the main country-to-country heatmap
    country_matrix = normalized_matrix[:, :-1]  # All columns except last (unmapped)
    sns.heatmap(
        country_matrix,
        xticklabels=country_labels,
        yticklabels=country_labels,
        annot=show_values,
        fmt='.1f',
        cmap=colormap,
        vmin=0,
        vmax=100,
        cbar=False,  # We'll add a shared colorbar later
        square=False,
        linewidths=0.5,
        linecolor='white',
        ax=ax1
    )
    
    # Create the unmapped dependencies heatmap (single column)
    unmapped_matrix = normalized_matrix[:, -1:] # Last column only
    sns.heatmap(
        unmapped_matrix,
        xticklabels=["Unmapped"],
        yticklabels=False,  # Don't repeat country labels
        annot=show_values,
        fmt='.1f',
        cmap=colormap,
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Dependency Percentage (%)'},
        square=False,
        linewidths=0.5,
        linecolor='white',
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
    # NOT all countries that appear in the dependency data
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
    
    # Build the dependency matrix (still using codes internally)
    # Any dependencies on countries NOT in country_codes_sorted will go to unmapped
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
        title = "Global Country Dependency Matrix"
        if args.vpn:
            title += f" (VPN: {args.vpn})"
    else:
        title = f"Country Dependency Matrix - {args.country}"
        if args.vpn:
            title += f" (VPN: {args.vpn})"
    
    # Generate output path if saving
    output_path = None
    if args.save:
        if args.all_countries:
            output_dir = Path("results/global_analysis")
            output_dir.mkdir(parents=True, exist_ok=True)
            vpn_suffix = f"_vpn_{args.vpn}" if args.vpn else ""
            output_path = output_dir / f"global_dependency_heatmap{vpn_suffix}.png"
        else:
            output_dir = Path(f"results/{args.code}/results/locality")
            output_dir.mkdir(parents=True, exist_ok=True)
            vpn_suffix = f"_vpn_{args.vpn}" if args.vpn else ""
            output_path = output_dir / f"dependency_heatmap{vpn_suffix}.png"
    
    # Create the heatmap with larger figure size for country names
    figsize = (16, 12) if args.all_countries else (12, 10)
    normalized_matrix = create_dependency_heatmap(
        matrix, 
        unmapped_dependencies, 
        country_labels,
        title=title,
        figsize=figsize,
        save_path=output_path,
        colormap=args.colormap,
        show_values=args.show_values
    )
    
    # Create summary plots only if flag is set
    if args.show_summary:
        create_summary_plots(normalized_matrix, country_labels)
    
    # Print comprehensive statistics
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