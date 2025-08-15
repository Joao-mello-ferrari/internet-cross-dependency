import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_classifications(file_path):
    """Load LLM predictions and ground truth classifications."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data['llm_predictions'], data['ground_truth']

def analyze_errors(llm_predictions, ground_truth):
    """Analyze classification errors by category."""
    # Get common URLs
    common_urls = set(llm_predictions.keys()) & set(ground_truth.keys())
    
    # Initialize counters
    category_stats = defaultdict(lambda: {
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'total_ground_truth': 0
    })
    
    # Count ground truth instances per category
    for url in common_urls:
        gt_category = ground_truth[url]
        category_stats[gt_category]['total_ground_truth'] += 1
    
    # Analyze predictions vs ground truth
    for url in common_urls:
        llm_pred = llm_predictions[url]
        gt_category = ground_truth[url]
        
        if llm_pred == gt_category:
            # Correct prediction
            category_stats[gt_category]['true_positives'] += 1
        else:
            # False negative for ground truth category
            category_stats[gt_category]['false_negatives'] += 1
            # False positive for predicted category
            category_stats[llm_pred]['false_positives'] += 1
    
    return category_stats

def create_error_histogram(category_stats):
    """Create histogram showing false positive rates by category."""
    categories = []
    error_counts = []
    error_percentages = []
    
    for category, stats in category_stats.items():
        # Skip the Unclassified category
        if category == "7. Unclassified":
            continue
            
        if stats['total_ground_truth'] > 0:  # Only include categories with ground truth instances
            false_positives = stats['false_positives']
            # Calculate false positive rate as percentage of ground truth instances for this category
            error_rate = (false_positives / stats['total_ground_truth']) * 100
            
            categories.append(category.replace(' & ', '\n& '))  # Break long names
            error_counts.append(false_positives)
            error_percentages.append(error_rate)
    
    # Sort by false positive count for better visualization
    sorted_data = sorted(zip(categories, error_counts, error_percentages), 
                        key=lambda x: x[1], reverse=True)
    categories, error_counts, error_percentages = zip(*sorted_data)
    
    # Create the plot with better size and spacing
    plt.figure(figsize=(12, 8))
    
    # Add padding to y-axis limits
    max_count = max(error_counts) if error_counts else 1
    y_padding = max_count * 0.15  # 15% padding
    
    bars = plt.bar(range(len(categories)), error_counts, alpha=0.7, color='lightcoral', 
                   width=0.6)  # Reduce bar width for better spacing
    
    # Add percentage labels on top of bars
    for i, (bar, percentage) in enumerate(zip(bars, error_percentages)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_padding*0.1,
                f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Set y-axis limits with padding
    plt.ylim(0, max_count + y_padding)
    
    plt.xlabel('Categories', fontsize=12, labelpad=10)
    plt.ylabel('Number of False Positives', fontsize=12, labelpad=10)
    plt.title('False Positive Classification Errors by Category\n(Percentage shows false positive rate relative to ground truth instances)', 
              fontsize=14, pad=20)
    plt.xticks(range(len(categories)), categories, rotation=45, ha='right', fontsize=10)
    
    # Add more spacing and padding
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)
    plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add some space between bars
    plt.margins(x=0.02)
    
    return plt

def print_detailed_stats(category_stats):
    """Print detailed statistics for each category."""
    print("\nDetailed False Positive Analysis:")
    print("=" * 90)
    print(f"{'Category':<40} {'GT':<4} {'FP':<4} {'FN':<4} {'FP Rate':<10}")
    print("=" * 90)
    
    for category, stats in sorted(category_stats.items(), 
                                 key=lambda x: x[1]['false_positives'], 
                                 reverse=True):
        # Skip the Unclassified category
        if category == "7. Unclassified":
            continue
            
        if stats['total_ground_truth'] > 0:
            fp_rate = (stats['false_positives'] / stats['total_ground_truth']) * 100
            
            print(f"{category:<40} {stats['total_ground_truth']:<4} "
                  f"{stats['false_positives']:<4} {stats['false_negatives']:<4} "
                  f"{fp_rate:<10.1f}%")

if __name__ == "__main__":
    # Load the data
    llm_data, ground_truth_data = load_classifications('/Users/joaomello/Desktop/tcc/classes.json')
    
    # Analyze errors
    category_stats = analyze_errors(llm_data, ground_truth_data)
    
    # Create histogram
    plt = create_error_histogram(category_stats)
    
    # Print detailed statistics
    print_detailed_stats(category_stats)
    
    # Save and show the plot
    plt.savefig('/Users/joaomello/Desktop/tcc/classification_errors.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nHistogram saved to: /Users/joaomello/Desktop/tcc/classification_errors.png")
    # Analyze errors
    category_stats = analyze_errors(llm_data, ground_truth_data)
    
    # Create histogram
    plt = create_error_histogram(category_stats)
    
    # Print detailed statistics
    print_detailed_stats(category_stats)
    
    # Save and show the plot
    plt.savefig('/Users/joaomello/Desktop/tcc/results/classification_errors.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nHistogram saved to: /Users/joaomello/Desktop/tcc/results/classification_errors.png")
else:
    print("Could not split the datasets. Please provide two separate JSON files.")
