import numpy as np
from pathlib import Path
import glob
import json

country_names = {
  "ag": "Antigua and Barbuda",
  "ar": "Argentina",
  "au": "Australia",
  "bb": "Barbados",
  "bo": "Bolivia",
  "br": "Brazil",
  "bs": "Bahamas",
  "bz": "Belize",
  "ca": "Canada",
  "cl": "Chile",
  "co": "Colombia",
  "cr": "Costa Rica",
  "cu": "Cuba",
  "de": "Germany",
  "dm": "Dominica",
  "do": "Dominican Republic",
  "ec": "Ecuador",
  "eg": "Egypt",
  "es": "Spain",
  "fr": "France",
  "gb": "United Kingdom",
  "gd": "Grenada",
  "gt": "Guatemala",
  "gy": "Guyana",
  "hn": "Honduras",
  "ht": "Haiti",
  "id": "Indonesia",
  "in": "India",
  "it": "Italy",
  "jm": "Jamaica",
  "jp": "Japan",
  "kn": "Saint Kitts and Nevis",
  "lc": "Saint Lucia",
  "mx": "Mexico",
  "ng": "Nigeria",
  "ni": "Nicaragua",
  "nz": "New Zealand",
  "pa": "Panama",
  "pe": "Peru",
  "pg": "Papua New Guinea",
  "py": "Paraguay",
  "sr": "Suriname",
  "sv": "El Salvador",
  "tt": "Trinidad and Tobago",
  "us": "United States",
  "uy": "Uruguay",
  "vc": "Saint Vincent and the Grenadines",
  "ve": "Venezuela",
  "za": "South Africa",
  "pt": "Portugal",
  "nl": "Netherlands", 
  "ie": "Ireland",
  "sg": "Singapore",
  "my": "Malaysia",
  "be": "Belgium",
}

country_codes = {
    "Antigua and Barbuda": "ag",
    "Argentina": "ar",
    "Australia": "au",
    "Barbados": "bb",
    "Bolivia": "bo",
    "Brazil": "br",
    "Bahamas": "bs",
    "Belize": "bz",
    "Canada": "ca",
    "Chile": "cl",
    "Colombia": "co",
    "Costa Rica": "cr",
    "Cuba": "cu",
    "Germany": "de",
    "Dominica": "dm",
    "Dominican Republic": "do",
    "Ecuador": "ec",
    "Egypt": "eg",
    "Spain": "es",
    "France": "fr",
    "United Kingdom": "gb",
    "Grenada": "gd",
    "Guatemala": "gt",
    "Guyana": "gy",
    "Honduras": "hn",
    "Haiti": "ht",
    "Indonesia": "id",
    "India": "in",
    "Italy": "it",
    "Jamaica": "jm",
    "Japan": "jp",
    "Saint Kitts and Nevis": "kn",
    "Saint Lucia": "lc",
    "Mexico": "mx",
    "Nigeria": "ng",
    "Nicaragua": "ni",
    "New Zealand": "nz",
    "Panama": "pa",
    "Peru": "pe",
    "Papua New Guinea": "pg",
    "Paraguay": "py",
    "Suriname": "sr",
    "El Salvador": "sv",
    "Trinidad and Tobago": "tt",
    "United States": "us",
    "Uruguay": "uy",
    "Saint Vincent and the Grenadines": "vc",
    "Venezuela": "ve",
    "South Africa": "za"
}

def get_country_name(country_code):
    """Returns the full name of a country given its code."""
    return country_names.get(country_code, country_code)

def get_country_name_mapping():
    """
    Get mapping from country codes to full country names.
    
    Returns:
        dict mapping country codes to country names
    """
    return country_names.copy()

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

def load_classified_websites(json_path):
    """
    Load classified websites from JSON file and create domain-to-class mapping.
    
    Args:
        json_path: Path to the classified_websites.json file
    
    Returns:
        dict mapping domains (without https://) to class names
    """
    with open(json_path, 'r') as f:
        classified_data = json.load(f)
    
    # Create mapping from domain (without protocol) to class
    domain_to_class = {}
    for url, class_name in classified_data.items():
        # Remove protocol and store domain
        domain = url.replace('https://', '').replace('http://', '')
        domain_to_class[domain] = class_name
    
    return domain_to_class

def get_class_mapping():
    """
    Get mapping from class names to numerical IDs.
    
    Returns:
        dict mapping class names to numbers (1-4)
    """
    return {
        "Critical Services": 1,
        "News": 2, 
        "General Digital Services": 3,
        "Entertainment": 4
    }

def process_experiment_country(location_data, locedge_data, dependency_counts, consider_anycast=False, class_filter=None, domain_to_class=None, add_anycast_separately=False, cdn_data=None):
    """
    Process a single experiment to count dependencies between countries.
    
    Args:
        location_data: dict mapping domains to their locations
        locedge_data: dict with additional location/edge data
        dependency_counts: nested dict to accumulate dependency counts
        class_filter: optional class name to filter domains by
        domain_to_class: optional dict mapping domains to their classes
        add_anycast_separately: whether to count anycast locations separately
    
    Returns:
        unknown_count: number of domains with unknown location
    """
    unknown_count = 0
    
    # unknown_count = 0
    # for domain, location in location_data.items():
    #     content_location = locedge_data.get(domain, {}).get("contentLocality")

    #     if location is None and content_location is None:
    #         unknown_count += 1
    #         continue

    #     if content_location:
    #         country_counts[content_location]["no_anycast"] += 1
    #     elif "- anycast" in location:
    #         raw_location = location.replace(" - anycast", "")
    #         country_counts[raw_location]["anycast"] += 1
    #     else:
    #         country_counts[location]["no_anycast"] += 1

    # return unknown_count

    for domain, location in location_data.items():
        if cdn_data is not None:
            cdn_location = cdn_data.get(domain)
            
            # If we provide cdn_data, skip domains that are not served using a CDN
            if cdn_location is None:
                continue


        # Apply class filter if provided
        if class_filter is not None and domain_to_class is not None:
            domain_class = domain_to_class.get(domain)
            if domain_class != class_filter:
                continue

        content_location = locedge_data.get(domain, {}).get("contentLocality")

        if location is None and content_location is None:
            unknown_count += 1
            continue
        
        # This is currently used by the bars_by_country script to consider anycast as a separate category
        if consider_anycast:
            if content_location:
                dependency_counts[content_location]["no_anycast"] += 1
            elif "- anycast" in location:
                raw_location = location.replace(" - anycast", "")
                dependency_counts[raw_location]["anycast"] += 1
            else:
                dependency_counts[location]["no_anycast"] += 1
            continue
        
        # This is currently used by the country_dependency_heatmap script to ignore anycast locations
        # Use content_location if available, otherwise use location
        final_location = content_location if content_location else location
        
        # Clean up location (remove anycast suffix if present)
        if "- anycast" in final_location:
            if add_anycast_separately:
                if "anycast" not in dependency_counts:
                    dependency_counts["anycast"] = 0
                dependency_counts["anycast"] += 1
            else:
                unknown_count += 1
            continue
        
        # Count this dependency
        if final_location not in dependency_counts:
            dependency_counts[final_location] = 0
        dependency_counts[final_location] += 1

    return unknown_count

def process_experiment_cdn(cdn_data, locedge_data, provider_data, provider_counts, class_filter=None, domain_to_class=None):
    """
    Process a single experiment to count dependencies on CDN providers.
    
    Args:
        cdn_data: dict mapping domains to their CDN providers
        locedge_data: dict with additional location/edge data
        provider_data: dict mapping domains to provider info from whois
        provider_counts: nested dict to accumulate provider counts
        class_filter: optional class name to filter domains by
        domain_to_class: optional dict mapping domains to their classes
    
    Returns:
        unknown_count: number of domains with unknown provider
    """
    unknown_count = 0
    for domain, cdn_string in cdn_data.items():
        # Apply class filter if provided
        if class_filter is not None and domain_to_class is not None:
            domain_class = domain_to_class.get(domain)
            if domain_class != class_filter:
                continue
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
                if provider == "amazon aws":
                    provider = "cloudfront"
                provider_counts[provider]["cdn"] += 1
        else:
            for provider in map(str.lower, no_cdn_providers):
                if provider == "aws (not cdn)":
                    continue
                if provider == "amazon aws":
                    provider = "cloudfront"
                provider_counts[provider]["no_cdn"] += 1

    return unknown_count

def aggregate_small_items(data_dict, threshold=5, others_label="others"):
    """
    Aggregate items with counts below threshold into an 'others' category.
    
    Args:
        data_dict: dict with items and their counts
        threshold: minimum count to keep item separate
        others_label: label for aggregated items
    
    Returns:
        tuple: (final_dict, others_count)
    """
    final_dict = {}
    others_count = 0
    others_items = 0
    
    for item, count in data_dict.items():
        if isinstance(count, dict):
            # Handle nested dictionaries (e.g., {"cdn": X, "no_cdn": Y})
            total = sum(count.values())
        else:
            total = count
            
        if total < threshold:
            if isinstance(count, dict):
                if others_label not in final_dict:
                    final_dict[others_label] = {k: 0 for k in count.keys()}
                for k, v in count.items():
                    final_dict[others_label][k] += v
            else:
                others_count += count
            others_items += 1
        else:
            final_dict[item] = count
    
    if isinstance(list(data_dict.values())[0], dict) and others_label in final_dict:
        # For nested dicts, the count is already in final_dict
        return final_dict, others_items
    elif others_count > 0:
        final_dict[f"{others_label} ({others_items})"] = others_count
        return final_dict, others_items
    else:
        return final_dict, others_items

def normalize_matrix(matrix, unmapped_dependencies=None):
    """
    Normalize a matrix so each row sums to 100%.
    
    Args:
        matrix: n x m matrix of dependencies
        unmapped_dependencies: optional array of unmapped dependencies to include
    
    Returns:
        normalized_matrix: normalized matrix with percentages
    """
    matrix = np.array(matrix, dtype=float)
    
    if unmapped_dependencies is not None:
        unmapped_dependencies = np.array(unmapped_dependencies, dtype=float)
        # Combine matrix with unmapped dependencies as last column
        extended_matrix = np.column_stack([matrix, unmapped_dependencies])
    else:
        extended_matrix = matrix
    
    # Calculate row sums
    row_sums = np.sum(extended_matrix, axis=1)
    
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    
    # Normalize each row to sum to 100%
    normalized_matrix = (extended_matrix / row_sums[:, np.newaxis]) * 100
    
    return normalized_matrix

def find_experiment_files(base_path, file_types, vpn_filter=None):
    """
    Find experiment files of specified types in the base path.
    
    Args:
        base_path: Path to search in
        file_types: list of file names to look for (e.g., ["location.json", "locedge.json"])
        vpn_filter: optional VPN filter string
    
    Returns:
        list of tuples with file paths for each experiment
    """
    if not file_types:
        return []
    
    # Find all instances of the first file type
    first_file_pattern = str(base_path / f"**/{file_types[0]}")
    experiment_paths = []
    
    for first_file in glob.glob(first_file_pattern, recursive=True):
        if vpn_filter and vpn_filter not in first_file:
            continue
        
        experiment_dir = Path(first_file).parent
        file_paths = [first_file]
        
        # Check if all other required files exist
        all_exist = True
        for file_type in file_types[1:]:
            file_path = experiment_dir / file_type
            if file_path.exists():
                file_paths.append(str(file_path))
            else:
                all_exist = False
                break
        
        if all_exist:
            experiment_paths.append(tuple(file_paths))
    
    return experiment_paths