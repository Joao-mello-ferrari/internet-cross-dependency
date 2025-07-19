import tldextract

def filter_unique_domains(urls, with_suffix=False):
    seen_domains = set()
    filtered_urls = []
    repeated_domains = {}

    for url in urls:
        # Extract the registered domain (e.g., google.com, example.com.br)
        extracted = tldextract.extract(url)
        main_domain = f"{extracted.domain}"
        if with_suffix:
            main_domain += f".{extracted.suffix}"

        if main_domain not in seen_domains:
            seen_domains.add(main_domain)
            filtered_urls.append(url)
        
        if main_domain in repeated_domains:
            repeated_domains[main_domain].append(url)
        else:
            repeated_domains[main_domain] = [url]
    
    keys_to_remove = [key for key in repeated_domains if len(repeated_domains[key]) == 1]
    for key in keys_to_remove:
        repeated_domains.pop(key)

    return filtered_urls, repeated_domains
