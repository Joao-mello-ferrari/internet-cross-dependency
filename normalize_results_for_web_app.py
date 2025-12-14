"""
CDN-Country Relationship Analyzer

Processes website classification data to generate CDN-country relationships
with intensity scores based on content criticality and geographic distribution.
"""

from collections import defaultdict
from glob import glob
from json import dumps, loads
from pathlib import Path

# Constants
CLASSES = ["Critical Services", "News", "General Digital Services", "Entertainment"]
CDNS = ["Cloudfront", "Cloudflare", "Fastly", "Akamai", "Google", "Yahoo"]
COUNTRIES = [
    "AR", "AU", "BR", "CA", "JP", "CO", "CR", "DE", "DO", "EG",
    "ES", "FR", "GB", "GT", "ID", "IN", "MX", "NG", "NZ", "PG",
    "IT", "US", "ZA"
]

INPUT_FILE = "classified_websites.json"
OUTPUT_FILE = "tcc_web.json"
RESULTS_DIR = "results"


def load_classified_websites(filepath):
    """Load the classified websites from JSON file."""
    with open(filepath) as f:
        return loads(f.read())


def sanitize_url(url):
    """Remove protocol prefix from URL."""
    return url.replace("https://", "").replace("http://", "")


def load_experiment_data(experiment_path):
    """Load CDN, location, and edge location data for an experiment."""
    experiment = Path(experiment_path)

    with open(experiment / "cdn.json") as f:
        cdn_data = loads(f.read())
    with open(experiment / "location.json") as f:
        location_data = loads(f.read())
    with open(experiment / "locedge.json") as f:
        loc_edge_data = loads(f.read())

    return cdn_data, location_data, loc_edge_data


def resolve_host_country(website, location_data, loc_edge_data):
    """Resolve the host country, handling anycast cases."""
    host_country = location_data.get(website)

    if host_country and "anycast" in host_country:
        host_country = loc_edge_data.get(website, {}).get("contentLocality")

    return host_country.upper() if host_country else None


def build_class_mapper(countries, classified_websites):
    """Build the mapping of class -> cdn -> origin_country -> host_country -> websites."""
    class_mapper = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for country in countries:
        base_path = Path(RESULTS_DIR) / country.lower()

        with open(base_path / "output.json") as f:
            sanitized_websites = [sanitize_url(site) for site in loads(f.read())]

        for experiment_path in glob(str(base_path / "locality" / "*")):
            cdn_data, location_data, loc_edge_data = load_experiment_data(experiment_path)

            for content_class in CLASSES:
                for website in sanitized_websites:
                    if classified_websites.get(website) != content_class:
                        continue
                    if cdn_data.get(website) is None:
                        continue

                    for cdn in CDNS:
                        if cdn.lower() in cdn_data[website].lower():
                            host_country = resolve_host_country(website, location_data, loc_edge_data)
                            if host_country:
                                class_mapper[content_class][cdn][country][host_country].append(website)

    return class_mapper


def calculate_totals(class_mapper):
    """Calculate normalization totals from the class mapper."""
    origin_totals = defaultdict(int)
    origin_class_totals = defaultdict(lambda: defaultdict(int))
    max_websites = 0

    for content_class in class_mapper:
        for cdn in class_mapper[content_class]:
            for origin_country in class_mapper[content_class][cdn]:
                for host_country in class_mapper[content_class][cdn][origin_country]:
                    count = len(class_mapper[content_class][cdn][origin_country][host_country])
                    origin_totals[origin_country] += count
                    origin_class_totals[origin_country][content_class] += count
                    max_websites = max(max_websites, count)

    return origin_totals, origin_class_totals, max_websites


def calculate_intensity(num_websites, origin_country, content_class, origin_totals, origin_class_totals):
    """
    Calculate intensity score based on:
    - Percentage of origin total (overall dependency)
    - Percentage within content class (class-specific dependency)
    - Class criticality bonus
    """
    pct_of_origin = (num_websites / origin_totals[origin_country] * 100
                     if origin_totals[origin_country] > 0 else 0)

    pct_of_class = (num_websites / origin_class_totals[origin_country][content_class] * 100
                    if origin_class_totals[origin_country][content_class] > 0 else 0)

    criticality_index = CLASSES.index(content_class)
    criticality_bonus = 30 / (criticality_index + 1)

    intensity = pct_of_origin + pct_of_class + criticality_bonus

    return max(0, min(100, round(intensity)))


def generate_relationships(class_mapper, origin_totals, origin_class_totals):
    """Generate the output relationship records."""
    relationships = []

    for content_class in class_mapper:
        for cdn in class_mapper[content_class]:
            for origin_country in class_mapper[content_class][cdn]:
                for host_country in class_mapper[content_class][cdn][origin_country]:
                    websites = class_mapper[content_class][cdn][origin_country][host_country]
                    num_websites = len(websites)

                    intensity = calculate_intensity(
                        num_websites, origin_country, content_class,
                        origin_totals, origin_class_totals
                    )

                    relationships.append({
                        "id": f"rel{len(relationships) + 1}",
                        "originCountry": origin_country,
                        "hostCountry": host_country,
                        "cdnProvider": cdn,
                        "protocol": {"type": "IPv4", "version": "4.0"},
                        "contentClass": content_class,
                        "intensity": intensity,
                        "numWebsites": num_websites,
                        "latency": None,
                        "bandwidth": None,
                        "reliability": None,
                    })

    return relationships


def main():
    """Main entry point."""
    classified_websites = load_classified_websites(INPUT_FILE)

    class_mapper = build_class_mapper(COUNTRIES, classified_websites)

    origin_totals, origin_class_totals, _ = calculate_totals(class_mapper)

    relationships = generate_relationships(class_mapper, origin_totals, origin_class_totals)

    with open(OUTPUT_FILE, "w") as f:
        f.write(dumps(relationships, indent=2))

    print(f"Generated {len(relationships)} relationships to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

