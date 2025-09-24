import psycopg2
import ipaddress
from os import getenv
from dotenv import load_dotenv

# Database connection
load_dotenv()
DB_NAME = getenv("POSTGRES_DB") or "geo_db"
DB_USER = getenv("POSTGRES_USER") or "postgres"
DB_PASS = getenv("POSTGRES_PASSWORD") or "postgres"
DB_HOST = getenv("POSTGRES_HOST") or "localhost"
DB_PORT = getenv("POSTGRES_PORT") or "5432"


def ip_to_int(ip):
    """Convert IP string to integer."""
    return int(ipaddress.IPv4Address(ip))

def get_country(ip):
    """Find the country for a given IP address."""
    try:
        ip_int = ip_to_int(ip)
    except Exception as e:
        return None
    
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()

    query = """SELECT find_location(%s);"""
    cur.execute(query, (ip_int,))
    result = cur.fetchone()
    
    cur.close()
    conn.close()
    
    return result[0].replace("(", "").replace(")", "").split(",") if result else None

def get_anycast(ip):
    """Find the country for a given IP address."""
    try:
        ip_int = ip_to_int(ip)
    except Exception as e:
        return None
    
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()
    
    query = """SELECT find_anycast(%s);"""

    cur.execute(query, (ip_int,))
    result = cur.fetchone()
    if result is None:
        return False
    
    return True if result[0] == "true" else False
    
def classify_provider(provider):
    if provider is None:
        return None
    
    # Define known cloud providers
    providers = ["amazon", "microsoft", "hostinger", "go-daddy", "azion", "cloudflarenet", "artfiles", "digitalocean", "akamai"]
    
    known_providers = {
        "amazon": "Amazon",
        "aws": "Amazon",
        "microsoft": "Microsoft",
        "azure": "Microsoft",
        "google": "Google",
        "gcp": "Google",
        "cloudflarenet": "Cloudflare",
        "cloudflare": "Cloudflare",
        "akamai": "Akamai",
        "oracle": "Oracle",
        "digitalocean": "DigitalOcean",
        "linode": "Linode",
        "hetzner": "Hetzner",
        "ovh": "OVH",
        "ibm": "IBM",
        "alibaba": "Alibaba",
        "taobao": "Alibaba",
        "artfiles": "Artfiles",
        "azion": "Azion",
        "go-daddy": "GoDaddy",
        "hostinger": "Hostinger",
    }
    for p in providers:
        if p in provider.lower():
            return known_providers.get(p)
    
        
    provider_lower = provider.lower().split(" ")[0]
    return known_providers.get(provider_lower, None)
