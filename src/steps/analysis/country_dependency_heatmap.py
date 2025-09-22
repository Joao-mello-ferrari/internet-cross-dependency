import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openchord_modified as ocd
import argparse
import json
from pathlib import Path
from collections import defaultdict

from src.steps.analysis.helpers import (
    get_all_country_codes, convert_codes_to_names, sort_countries_by_continent,
    get_continent_mapping, process_experiment_country, normalize_matrix, find_experiment_files
)

# ===================
# Argument Parser
# ===================
def parse_arguments():
    parser = argparse.ArgumentParser(description='Create country dependency heatmap and chord visualizations')
    parser.add_argument("--country", type=str, help="Country label (for single country analysis)")
    parser.add_argument("--code", type=str, help="Country code used for folder path (for single country analysis)")
    parser.add_argument("--all-countries", action="store_true", help="Process all countries to create comprehensive matrix")
    parser.add_argument("--vpn", type=str, default=None, help="Optional VPN country code to filter")
    parser.add_argument('--show-summary', action='store_true', help='Show additional summary plots')
    parser.add_argument("--save", action="store_true", help="Save the figures")
    parser.add_argument("--heatmap-only", action="store_true", help="Generate only heatmap (skip chord diagram)")
    parser.add_argument("--chord-only", action="store_true", help="Generate only chord diagram (skip heatmap)")
    return parser.parse_args()

# ===================
# Country Matrix Functions
# ===================
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

def create_dependency_chord(matrix, unmapped_dependencies, country_labels, save_path=None):
    """
    Create a dependency chord diagram visualization.
    
    Args:
        matrix: n x n matrix of dependencies between countries
        country_labels: list of country names/codes
        save_path: path to save the figure (optional)
    """
    
    # Create a copy to avoid modifying the original matrix
    # Concatenate the unmapped dependencies as a last column
    matrix = np.concatenate([matrix, unmapped_dependencies[:, np.newaxis]], axis=1)
    # Normalize each row to sum to 100%
    row_sums = matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    matrix_processed = ((matrix / row_sums) * 100)[:, :-1]  # Exclude last column (unmapped) for chord diagram

    # Make matrix symmetric by adding small values to zero cells in transpose positions
    for n in range(matrix_processed.shape[0]):
        for m in range(matrix_processed.shape[1]):
            # If cell (n,m) has a value and cell (m,n) is zero, add small value to (m,n)
            if matrix_processed[n, m] != 0 and matrix_processed[m, n] == 0:
                matrix_processed[m, n] = matrix_processed[n, m] * 0.01
    
    # Calculate column sums and apply cubic root transformation for gentle scaling
    col_sums = np.sum(matrix_processed, axis=0)
    col_sums_cbrt = np.sqrt(col_sums + 1e-10)  # Add tiny value for numerical stability

    # Apply scaling
    matrix_scaled = matrix_processed * col_sums_cbrt[:, np.newaxis]
    
    # Create and display the chord diagram
    print(f"Creating chord diagram with {len(country_labels)} countries...")
    print(f"Original column sums: {col_sums}")
    print(f"Scaled column sums: {col_sums_cbrt}")
    
    fig = ocd.Chord(matrix_scaled, country_labels, scale=list(col_sums_cbrt))
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
    Main function that processes locality data and creates country dependency visualizations
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
        file_types = ["location.json", "locedge.json"]
        experiment_paths = find_experiment_files(base_path, file_types, args.vpn)

        print(f"  Found {len(experiment_paths)} experiments")
        
        # Process all experiments for this country
        for location_path, locedge_path in experiment_paths:
            with open(location_path) as f:
                location_data = json.load(f)
            with open(locedge_path) as f:
                locedge_data = json.load(f)

            # Process this experiment to get dependency counts
            dependency_counts = defaultdict(int)
            unknown_count = process_experiment_country(location_data, locedge_data, dependency_counts)
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
            unmapped_dependencies,
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