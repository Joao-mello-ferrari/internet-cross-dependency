import math
import re
from src.lib import run_command

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def find_ipv6_has_address(hostname):
    try:
        results = run_command(["dig", "+short", hostname, "AAAA"])
        # Use regex to extract the first IPv6 address from the results
        ipv6_pattern = r'\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}|\b(?:[0-9a-fA-F]{1,4}:){1,7}:|\b:(:[0-9a-fA-F]{1,4}){1,7}\b'
        for result in results.splitlines():
            _match = re.search(ipv6_pattern, result)
            if _match:
                return True
    except: pass            
    return False

country_name_mapper = {
  "ag": "AntiguaandBarbuda",
  "ar": "Argentina",
  "bs": "Bahamas",
  "bb": "Barbados",
  "bz": "Belize",
  "bo": "Bolivia",
  "br": "Brazil",
  "ca": "Canada",
  "cl": "Chile",
  "co": "Colombia",
  "cr": "CostaRica",
  "cu": "Cuba",
  "dm": "Dominica",
  "do": "DominicanRepublic",
  "ec": "Ecuador",
  "sv": "ElSalvador",
  "gd": "Grenada",
  "gt": "Guatemala",
  "gy": "Guyana",
  "ht": "Haiti",
  "hn": "Honduras",
  "jm": "Jamaica",
  "mx": "Mexico",
  "ni": "Nicaragua",
  "pa": "Panama",
  "py": "Paraguay",
  "pe": "Peru",
  "kn": "SaintKittsandNevis",
  "lc": "SaintLucia",
  "vc": "SaintVincentandtheGrenadines",
  "sr": "Suriname",
  "tt": "TrinidadandTobago",
  "us": "UnitedStates",
  "uy": "Uruguay",
  "ve": "Venezuela"
}