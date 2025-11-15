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
    get_continent_mapping, process_experiment_country, normalize_matrix, find_experiment_files,
    load_classified_websites, get_class_mapping, country_codes, country_names
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
    parser.add_argument("--show-plot", action="store_true", help="Show the plots interactively")
    parser.add_argument("--class", type=int, choices=[1, 2, 3, 4], help="Filter analysis by website class (1=Critical Services, 2=News, 3=General Digital Services, 4=Entertainment)")
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
                if target_country == "anycast":
                    # Skip anycast entries in this matrix
                    continue
                if target_country in country_to_idx:
                    # Dependency on a country we studied (has folder in results/)
                    target_idx = country_to_idx[target_country]
                    matrix[source_idx, target_idx] = count
                else:
                    # Dependency on a country we did NOT study (unmapped)
                    unmapped_dependencies[source_idx] += count
    
    return matrix, unmapped_dependencies

def create_dependency_heatmap(matrix, unmapped_dependencies, country_labels, 
                            title="Country Dependency Matrix", save_path=None, show_plot=False):
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
    if show_plot:
        fig.show()
    
    return normalized_matrix

def create_dependency_chord(matrix, unmapped_dependencies, country_labels, save_path=None, show_plot=False):
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
    if show_plot:
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

def get_stratified_country_data(country_codes, args, domain_to_class):
    """
    Process experiment data to get country dependency counts stratified by content categories.
    
    Args:
        country_codes: list of country codes to process
        args: command line arguments
        domain_to_class: mapping from domain to content class
    
    Returns:
        country_class_data: dict of {target_country: {class_name: count}}
        anycast_class_data: dict of {class_name: count} for anycast
        unmapped_class_data: dict of {class_name: count} for unmapped
    """
    # Get class mapping
    class_mapping = get_class_mapping()
    
    # Initialize data structures
    country_class_data = defaultdict(lambda: defaultdict(int))
    anycast_class_data = defaultdict(int)
    unmapped_class_data = defaultdict(int)
    
    total_experiments = 0
    
    for country_code in country_codes:
        print(f"\nProcessing country for stratification: {country_code}")
        base_path = Path(f"results/{country_code}/locality")

        # Find all experiment files for this country
        file_types = ["location.json", "locedge.json", "cdn.json"]
        experiment_paths = find_experiment_files(base_path, file_types, args.vpn)

        print(f"  Found {len(experiment_paths)} experiments")
        
        # Process each content class separately
        for class_name, _ in class_mapping.items():
            for location_path, locedge_path, cdn_path in experiment_paths:
                with open(location_path) as f:
                    location_data = json.load(f)
                with open(locedge_path) as f:
                    locedge_data = json.load(f)
                with open(cdn_path) as f:
                    cdn_data = json.load(f)

                # Process this experiment with class filter
                dependency_counts = defaultdict(int)
                unknown_count = process_experiment_country(
                    location_data, locedge_data, dependency_counts, 
                    False, class_name, domain_to_class, True, cdn_data
                )

                # Add to stratified data
                for target_country, count in dependency_counts.items():
                    if target_country == "anycast":
                        anycast_class_data[class_name] += count
                    elif target_country in country_codes:
                        country_class_data[target_country][class_name] += count
                    else:
                        # Unmapped country
                        unmapped_class_data[class_name] += count
        
        total_experiments += len(experiment_paths)
    
    print(f"Processed {total_experiments} total experiments for stratification")
    return dict(country_class_data), dict(anycast_class_data), dict(unmapped_class_data)

def create_stratified_country_barplot(country_class_data, anycast_class_data, unmapped_class_data, 
                                    country_labels, title="Country Dependencies by Content Category", 
                                    save_path=None, show_plot=False):
    """
    Create a stratified bar plot showing country dependencies broken down by content categories.
    
    Args:
        country_class_data: dict of {country: {class_name: count}}
        anycast_class_data: dict of {class_name: count} for anycast
        unmapped_class_data: dict of {class_name: count} for unmapped
        country_labels: list of country names
        title: plot title
        save_path: path to save the figure (optional)
        show_plot: whether to show the plot interactively
    """
    # Get class mapping for consistent ordering
    class_mapping = get_class_mapping()
    class_names = list(class_mapping.keys())

    # Calculate totals for each country
    country_totals = {}
    for country in country_labels:
        code = country_codes.get(country)
        if code in country_class_data:
            country_totals[country] = sum(country_class_data[code].values())
        else:
            country_totals[country] = 0
    
    # Add anycast and unmapped totals
    anycast_total = sum(anycast_class_data.values())
    unmapped_total = sum(unmapped_class_data.values())
    
    # Filter and sort countries by total count
    countries_with_data = {c: total for c, total in country_totals.items() if total > 0}
    countries_sorted = dict(sorted(countries_with_data.items(), key=lambda x: x[1], reverse=True))
    
    # Group small countries into "Others"
    min_threshold = sum(countries_sorted.values()) * 0.03  # Minimum count to show separately
    countries_to_show = {c: total for c, total in countries_sorted.items() if total >= min_threshold}

    others_data = defaultdict(int)
    for code, class_counts in country_class_data.items():
        if country_names.get(code) not in countries_to_show:
            for class_name, count in class_counts.items():
                others_data[class_name] += count
    
    # Prepare final data for plotting
    all_items = list(countries_to_show.keys())
    
    # Add special categories at the end
    if sum(others_data.values()) > 0:
        all_items.append("Others")
    if anycast_total > 0:
        all_items.append("Anycast") 
    if unmapped_total > 0:
        all_items.append("Unmapped")
    
    if not all_items:
        print("No data to plot")
        return [], []
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(max(12, len(all_items) * 1.2), 6))
    
    # Colors for each content category (same as CDN analysis)
    class_colors = {
        'Critical Services': '#d62728',     # Red
        'News': '#ff7f0e',                  # Orange  
        'General Digital Services': '#2ca02c',  # Green
        'Entertainment': '#1f77b4'          # Blue
    }
    
    # Bar width and positioning
    bar_width = 0.8 / len(class_names)
    x_positions = np.arange(len(all_items))
    
    # Create bars for each class
    for i, class_name in enumerate(class_names):
        class_counts = []
        for item in all_items:
            if item == "Others":
                count = others_data.get(class_name, 0)
            elif item == "Anycast":
                count = anycast_class_data.get(class_name, 0)
            elif item == "Unmapped":
                count = unmapped_class_data.get(class_name, 0)
            else:
                count = country_class_data.get(country_codes.get(item), {}).get(class_name, 0)
            class_counts.append(count)
        
        # Calculate total for this class across all items for percentage calculation
        class_total = sum(class_counts)
        
        # Position bars for this class
        x_pos = x_positions + (i - len(class_names)/2 + 0.5) * bar_width
        
        bars = ax.bar(x_pos, class_counts, bar_width, 
                     label=class_name, 
                     color=class_colors.get(class_name, '#808080'),
                     alpha=0.8)
        
        # Add percentage labels on bars (only for non-zero values)
        for bar, count in zip(bars, class_counts):
            if count > 0 and class_total > 0:
                percentage = (count / class_total) * 100
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(class_counts) * 0.01,
                       f'{percentage:.1f}%', ha='center', va='bottom', fontsize=5)
    
    # Customize the plot
    ax.set_xlabel('Origins', fontsize=14)
    ax.set_ylabel('Amount of Websites', fontsize=14)
    #ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([item.title() for item in all_items], rotation=45, ha='right')
    ax.legend(title='Content Categories', loc='upper center')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Stratified country dependency bar plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return all_items, class_names

def create_average_dependency_barplot(matrix, unmapped_dependencies, country_labels, 
                                    title="Average Country Dependency", save_path=None, all_dependency_data=None, show_plot=False):
    """
    Create a bar plot showing average dependency percentage for each country.
    
    Args:
        matrix: n x n matrix of dependencies between countries (absolute counts)
        unmapped_dependencies: array of unmapped dependencies (absolute counts)
        country_labels: list of country names
        title: plot title
        save_path: path to save the figure (optional)
        all_dependency_data: original dependency data including anycast (optional)
    """
    
    # Calculate total dependencies for each target country (column sums)
    country_totals = np.sum(matrix, axis=0)  # Sum dependencies TO each country
    unmapped_total = np.sum(unmapped_dependencies)  # Total unmapped dependencies
    
    # Extract anycast data if available
    anycast_total = 0
    if all_dependency_data:
        for country_data in all_dependency_data.values():
            if "anycast" in country_data:
                anycast_total += country_data["anycast"]
    
    # Calculate global total for percentage calculation
    total_global_dependencies = np.sum(country_totals) + unmapped_total + anycast_total
    
    # Calculate percentages
    country_percentages = country_totals / total_global_dependencies * 100
    unmapped_percentage = unmapped_total / total_global_dependencies * 100
    anycast_percentage = anycast_total / total_global_dependencies * 100
    
    # Combine data for plotting (include anycast)
    all_labels = country_labels + ['Anycast', 'Unmapped']
    all_percentages = np.concatenate([country_percentages, [anycast_percentage], [unmapped_percentage]])
    all_totals = np.concatenate([country_totals, [anycast_total], [unmapped_total]])

    # Filter out zero values
    non_zero_mask = all_totals > 0
    filtered_labels = [label for i, label in enumerate(all_labels) if non_zero_mask[i]]
    filtered_percentages = all_percentages[non_zero_mask]
    filtered_totals = all_totals[non_zero_mask]
    
    # Sort by percentage (descending), but keep Anycast and Unmapped at the end if present
    special_items = {}  # Store special items to add at the end
    
    # Extract special items (Anycast and Unmapped)
    for special_label in ['Anycast', 'Unmapped']:
        if special_label in filtered_labels:
            idx = filtered_labels.index(special_label)
            special_items[special_label] = {
                'percentage': filtered_percentages[idx],
                'total': filtered_totals[idx]
            }
    
    # Remove special items from main lists for sorting
    temp_labels = [label for label in filtered_labels if label not in ['Anycast', 'Unmapped']]
    temp_percentages = [pct for i, pct in enumerate(filtered_percentages) 
                       if filtered_labels[i] not in ['Anycast', 'Unmapped']]
    temp_totals = [total for i, total in enumerate(filtered_totals) 
                  if filtered_labels[i] not in ['Anycast', 'Unmapped']]
    
    # Sort the regular country data
    if temp_labels:
        sorted_data = sorted(zip(temp_percentages, temp_labels, temp_totals), reverse=True)
        sorted_percentages, sorted_labels, sorted_totals = zip(*sorted_data)
        sorted_percentages = list(sorted_percentages)
        sorted_labels = list(sorted_labels)
        sorted_totals = list(sorted_totals)
    else:
        sorted_percentages, sorted_labels, sorted_totals = [], [], []
    
    # Add special items at the end (Anycast before Unmapped)
    for special_label in ['Anycast', 'Unmapped']:
        if special_label in special_items and special_items[special_label]['total'] > 0:
            sorted_percentages.append(special_items[special_label]['percentage'])
            sorted_labels.append(special_label)
            sorted_totals.append(special_items[special_label]['total'])

    sorted_percentages_grouped = []
    sorted_labels_grouped = []
    sorted_totals_grouped = []
    others_percentage = 0
    others_label = "Others"
    others_total = 0
    
    for (perc, label, total) in zip(sorted_percentages, sorted_labels, sorted_totals):
        if perc < 3.0 and label not in ['Anycast', 'Unmapped']:
            others_percentage += perc
            others_total += total
            continue
        # Group small percentages into 'Others' if needed (optional)
        sorted_percentages_grouped.append(perc)
        sorted_labels_grouped.append(label)
        sorted_totals_grouped.append(total)
    
    if others_percentage > 0:
        # Find the position to insert 'Others' - before Anycast and Unmapped
        insert_pos = len(sorted_percentages_grouped)
        for i, label in enumerate(sorted_labels_grouped):
            if label in ['Anycast', 'Unmapped']:
                insert_pos = i
                break
        
        sorted_percentages_grouped.insert(insert_pos, others_percentage)
        sorted_labels_grouped.insert(insert_pos, others_label)
        sorted_totals_grouped.insert(insert_pos, others_total)

    if not sorted_labels:
        print("No data to plot")
        return [], []
    
    # Create the plot
    plt.figure(figsize=(max(12, len(sorted_labels_grouped) * 0.8), 5))
    
    # Create color map - different colors for special categories
    colors = []
    for label in sorted_labels_grouped:
        if label == 'Unmapped':
            colors.append("#e27b1c")  # red for unmapped
        elif label == 'Anycast':
            colors.append("#3a3ae3")  # purple for anycast
        elif label == 'Others':
            colors.append('#E9E911')  # orange for others
        else:
            colors.append('#690398')  # blue for countries

    bars = plt.bar(range(len(sorted_labels_grouped)), sorted_totals_grouped, color=colors)

    # Customize the plot
    plt.ylabel('Amount of Websites', fontsize=18)
    plt.ylim(0, max(sorted_totals_grouped) * 1.1)

    # Set x-axis labels
    plt.xticks(range(len(sorted_labels_grouped)), sorted_labels_grouped, rotation=45, ha='right', fontsize=15)
    plt.yticks(fontsize=15)
    
    # Add percentage labels on top of bars
    for bar, pct in zip(bars, sorted_percentages_grouped):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sorted_totals_grouped) * 0.01,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=13)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#690398', label='Host Countries'),
        Patch(facecolor='#E9E911', label='Grouped Host Countries'),
        Patch(facecolor='#3a3ae3', label='Anycast Host Countries'),
        Patch(facecolor='#e27b1c', label='Unmapped Host Countries')
    ]
    plt.legend(handles=legend_elements, loc='upper center')
    
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Average dependency bar plot saved to: {save_path}")
    
    if show_plot:
        plt.show()

    return sorted_percentages_grouped, sorted_labels_grouped

# ===================
# Main Function
# ===================
def main():
    """
    Main function that processes locality data and creates country dependency visualizations
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

    # Process all countries to build comprehensive dependency data
    all_dependency_data = defaultdict(lambda: defaultdict(int))
    total_unknown = 0
    total_experiments = 0

    for country_code in country_codes:
        print(f"\nProcessing country: {country_code}")
        base_path = Path(f"results/{country_code}/locality")

        # Find all experiment files for this country
        file_types = ["location.json", "locedge.json", "cdn.json"]
        experiment_paths = find_experiment_files(base_path, file_types, args.vpn)

        print(f"  Found {len(experiment_paths)} experiments")
        
        # Process all experiments for this country
        for location_path, locedge_path, cdn_path in experiment_paths:
            with open(location_path) as f:
                location_data = json.load(f)
            with open(locedge_path) as f:
                locedge_data = json.load(f)
            with open(cdn_path) as f:
                cdn_data = json.load(f)

            # Process this experiment to get dependency counts
            dependency_counts = defaultdict(int)
            unknown_count = process_experiment_country(location_data, locedge_data, dependency_counts, False, class_filter, domain_to_class, True, cdn_data)
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
    else:
        country_codes_sorted = [args.code]  # Single country analysis
    country_labels = convert_codes_to_names(country_codes_sorted)  # Convert to full names
    
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
        if class_filter:
            title += f" - Class {getattr(args, 'class')}: {class_filter}"
        if args.vpn:
            title += f" (VPN: {args.vpn})"
    else:
        title = f"Country Dependency Analysis - {args.country}"
        if class_filter:
            title += f" - Class {getattr(args, 'class')}: {class_filter}"
        if args.vpn:
            title += f" (VPN: {args.vpn})"
    
    # Generate output paths if saving
    heatmap_path = None
    chord_path = None
    barplot_path = None
    
    if args.save:
        if args.all_countries:
            output_dir = Path("results")
            output_dir.mkdir(parents=True, exist_ok=True)
            class_suffix = f"_class_{getattr(args, 'class')}" if class_filter else ""
            vpn_suffix = f"_vpn_{args.vpn}" if args.vpn else ""
            heatmap_path = output_dir / f"global_dependency_heatmap{class_suffix}{vpn_suffix}.png"
            chord_path = output_dir / f"global_dependency_chord{class_suffix}{vpn_suffix}.svg"
            barplot_path = output_dir / f"global_dependency_barplot{class_suffix}{vpn_suffix}.png"
        else:
            output_dir = Path(f"results/{args.code}/results/locality")
            output_dir.mkdir(parents=True, exist_ok=True)
            class_suffix = f"_class_{getattr(args, 'class')}" if class_filter else ""
            vpn_suffix = f"_vpn_{args.vpn}" if args.vpn else ""
            heatmap_path = output_dir / f"dependency_heatmap{class_suffix}{vpn_suffix}.png"
            chord_path = output_dir / f"dependency_chord{class_suffix}{vpn_suffix}.svg"
            barplot_path = output_dir / f"dependency_barplot{class_suffix}{vpn_suffix}.png"
    
    # Create visualizations based on arguments
    normalized_matrix = None
    
    if not args.chord_only:
        print("\n=== Creating Dependency Heatmap ===")
        normalized_matrix = create_dependency_heatmap(
            matrix, 
            unmapped_dependencies, 
            country_labels,
            title=title + " - Heatmap",
            save_path=heatmap_path,
            show_plot=args.show_plot
        )
    
    if not args.heatmap_only:
        print("\n=== Creating Dependency Chord Diagram ===")
        create_dependency_chord(
            matrix,
            unmapped_dependencies,
            country_labels,
            save_path=chord_path,
            show_plot=args.show_plot
        )
    
    # Create average dependency bar plot
    print("\n=== Creating Average Dependency Bar Plot ===")
    barplot_save_path = barplot_path if args.save else None
    
    # Choose between stratified and regular bar plot
    if class_filter is None:
        # No class filter - create stratified bar plot showing all content categories
        print("Creating stratified bar plot by content categories...")
        
        # Load classified websites data for stratification (if not already loaded)
        if domain_to_class is None:
            try:
                domain_to_class = load_classified_websites("classified_websites.json")
                print(f"Loaded {len(domain_to_class)} classified domains for stratification")
            except FileNotFoundError:
                print("Warning: classified_websites.json not found. Falling back to regular bar plot.")
                domain_to_class = None
            except Exception as e:
                print(f"Error loading classified websites: {e}. Falling back to regular bar plot.")
                domain_to_class = None
        
        if domain_to_class is not None:
            try:
                # Get stratified data
                country_class_data, anycast_class_data, unmapped_class_data = get_stratified_country_data(
                    country_codes, args, domain_to_class
                )
                
                # Create stratified bar plot
                stratified_title = "Country Dependencies by Content Category"
                if class_filter:
                    stratified_title += f" - Class {getattr(args, 'class')}: {class_filter}"
                if args.vpn:
                    stratified_title += f" (VPN: {args.vpn})"
                
                all_items, class_names = create_stratified_country_barplot(
                    country_class_data,
                    anycast_class_data,
                    unmapped_class_data,
                    country_labels,
                    title=stratified_title,
                    save_path=barplot_save_path,
                    show_plot=args.show_plot
                )
                
            except Exception as e:
                print(f"Error creating stratified plot: {e}. Falling back to regular bar plot.")
                # Fall back to regular bar plot
                avg_percentages, sorted_labels = create_average_dependency_barplot(
                    matrix,
                    unmapped_dependencies,
                    country_labels,
                    title=title + " - Average Dependencies",
                    save_path=barplot_save_path,
                    all_dependency_data=all_dependency_data,
                    show_plot=args.show_plot
                )
        else:
            # Fall back to regular bar plot
            avg_percentages, sorted_labels = create_average_dependency_barplot(
                matrix,
                unmapped_dependencies,
                country_labels,
                title=title + " - Average Dependencies",
                save_path=barplot_save_path,
                all_dependency_data=all_dependency_data,
                show_plot=args.show_plot
            )
    else:
        # Class filter applied - create regular bar plot
        avg_percentages, sorted_labels = create_average_dependency_barplot(
            matrix,
            unmapped_dependencies,
            country_labels,
            title=title + " - Average Dependencies",
            save_path=barplot_save_path,
            all_dependency_data=all_dependency_data,
            show_plot=args.show_plot
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