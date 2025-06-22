import json
import re
import argparse
import os
import httpx
from openai import OpenAI
from math import ceil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from os import getenv
from dotenv import load_dotenv


# =======================
# OPENAI CLIENT
# =======================
load_dotenv()
client = OpenAI(
    api_key=getenv("OPENAI_API_KEY"),
    http_client=httpx.Client(verify=False)
)

# =======================
# CLI ARGUMENTS
# =======================
parser = argparse.ArgumentParser(description='Classify website categories.')
parser.add_argument('input', type=str, help='Input JSON file with list of websites')
parser.add_argument('output', type=str, help='Output JSON file for results')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size (default: 10)')
parser.add_argument('--workers', type=int, default=10, help='Number of concurrent threads (default: 10)')
args = parser.parse_args()


# =======================
# LOAD INPUT DATA
# =======================
with open(args.input, 'r') as f:
    _websites = json.load(f)

websites = []
keys = []
for key in list(_websites.keys()):
    keys.append(key)
    websites.extend(_websites[key])    
        

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
- The categories are ordered by priority, so if a website fits multiple categories, choose the highest priority one.
- If there is no clear category, classify it as "13. Other".
- If the website is inaccessible, classify it as "14. Unknown".

Return ONLY a JSON mapping of each website to its category in this format:

{
  "example.com": "3. Entertainment & Leisure",
  "another.com": "4. Financial Services"
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
def classify_batch(batch, idx, keys):
    prompt = build_prompt(batch)
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            tools=[{"type": "web_search_preview"}],
            temperature=0
        )
        parsed = extract_json_from_text(response.output_text)
        #parsed = {
        #    "example.com": "3. Entertainment & Leisure",
        #    "another.com": "4. Financial Services"
        #}  # Mocked response for testing purposes
        return { keys[idx]: parsed } if parsed else {}
    except Exception as e:
        print(f"Error processing batch {batch}: {e}")
        return {}


# =======================
# PREPARE BATCHES
# =======================
batch_size = args.batch_size
batches = [
    websites[i * batch_size: (i + 1) * batch_size]
    for i in range(ceil(len(websites) / batch_size))
]


# =======================
# PROCESS IN PARALLEL
# =======================
results = {}

with ThreadPoolExecutor(max_workers=args.workers) as executor:
    future_to_batch = {executor.submit(classify_batch, batch, idx, keys): batch for idx, batch in enumerate(batches)}

    for future in tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc="Processing"):
        batch_result = future.result()
        if batch_result:
            results.update(batch_result)


# =======================
# SAVE OUTPUT
# =======================
os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Classification complete. Saved {len(results)} results to {args.output}")
