import json
import re
from openai import OpenAI

# =======================
# BUILD PROMPT
# =======================
def build_prompt(batch):
    return """You are a website classification agent.

You need to search the web to fetch the website content.
Your task is to classify the MAIN PURPOSE of the following websites into EXACTLY ONE of these categories:

1. Critical Public Government Services — Government and other essential public services (e.g., government portals).
2. Education & Study Services — education and related services (e.g., online schools, universities, courses, libraries, scientific databases, encyclopedias, educational content repositories).
3. Health — Health services (e.g. hospitals, clinics, pharmacies, health insurance, medical services, health information).
4. Financial Services — Banks, investment firms, insurance, fintech, cryptocurrency, and financial institutions.
5. Commerce & Retail — E-commerce platforms, online stores, marketplaces, product retailers.
6. Media & News — Journalism, newspapers, TV networks, radio, online news, and media outlets.
7. Entertainment, Social Media & Lifestyle — Streaming services, video platforms, recreational content, influencers, wellness, food, home, fashion.
8. Industry & Business Services — Industrial companies, B2B services, manufacturing, logistics, professional services, jobs & careers platforms.
9. Technology & Cloud Services — SaaS, cloud services, developer tools, hosting providers, CDN, APIs, and IT service companies.
10. Travel & Mobility — Airlines, hotels, tourism services, transportation, booking platforms.
11. Betting - Gambling, online casinos, betting platforms, and related services.
12. Adult Content - Websites primarily focused on adult content
13. Other — Any website that does not fit into the above categories.
14. Unknown — If you cannot access the website, classify it as "Unknown".

❗IMPORTANT:
- DO NOT classify based on the URL name. Analyze the page content and purpose via web search.
- Choose the category that BEST represents the PRIMARY function of the website.
- If the website covers multiple topics, pick the one that appears most prominently.
- If there is no clear category, classify it as "13. Other".
- If the website is inaccessible, classify it as "14. Unknown".

Return ONLY a JSON mapping of each website to its category. The keys should be exactly the website.

{
  "http://example.com": "3. Entertainment & Leisure",
  "https://another.com": "4. Financial Services"
}

Websites:
""" + "\n".join(f"- {site}" for site in batch)


# =======================
# JSON EXTRACTOR
# =======================
def extract_json_from_text(text):
    try:
        print(text)
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            print("No JSON found in text.")
            return None
        return json.loads(json_match.group(0))
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None


# =======================
# FUNCTION FOR API CALL
# =======================
def classify_batch(batch, client: OpenAI):
    prompt = build_prompt(batch)
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            tools=[{"type": "web_search_preview"}],
            temperature=0
        )
        parsed = extract_json_from_text(response.output_text)
        return parsed if parsed else {}
    except Exception as e:
        print(f"Error processing batch {batch}: {e}")
        return {}


def narrow_classes(classified_websites: object):
    mapping = {
        "1. Critical Public Government Services": "1. Critical & Social Services",
        "2. Education & Study Services": "1. Critical & Social Services", 
        "3. Health": "1. Critical & Social Services", 
        "4. Financial Services": "2. Financial Services",
        "5. Commerce & Retail": "3. Commerce, Retail & Industry",
        "6. Media & News": "4. Media & News",
        "7. Entertainment, Social Media & Lifestyle": "5. Entertainment & Social Media",
        "8. Industry & Business Services": "3. Commerce, Retail & Industry",
        "9. Technology & Cloud Services": "3. Commerce, Retail & Industry",
        "10. Travel & Mobility": "6. Travel & Mobility",
        "11. Betting": "5. Entertainment & Social Media",
        "12. Adult Content": "5. Entertainment & Social Media",
        "13. Other": "7. Unclassified",
        "14. Unknown": "7. Unclassified",
    }

    for site, category in classified_websites.items():
        if category not in mapping.keys():
            classified_websites[site] = "7. Unclassified"
        else:
            classified_websites[site] = mapping[category]
    return classified_websites