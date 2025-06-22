import argparse
from lib import run_command
from json import loads
import os
import json

def get_base_latency():
    my_location = loads(run_command(["curl", "https://ipinfo.io"]))
    return my_location.get("loc").split(","), my_location.get("country").lower()

def run_scripts(domain, base_latency, geopoint, fetch_websites, agg_func, vpn):
    if fetch_websites:
        #run_script(["make", "fetch_websites", f"COUNTRY=\"{args.country}\"", f"CODE=\"{args.code}\"", "QUERY=\"country\"", f"SEMESTER=\"{args.semester}\"", f"AMOUNT=\"{args.count}\""])
        pass
    
    #run_script(["make", "ping", f"COUNTRY=\"{args.country}\"", f"CODE=\"{args.code}\"", f"FETCH_FRESH_PROBES=\"--fetch_fresh_probes\""])
    #run_script(["make", "findcdn", f"COUNTRY=\"{args.country}\"", f"CODE=\"{args.code}\"", f"VPN=\"{vpn}\""])
    #run_script(["make", "geolocate", f"COUNTRY=\"{args.country}\"", f"CODE=\"{args.code}\"", f"VPN=\"{vpn}\""])
    #run_script(["make", "locedge", f"COUNTRY=\"{args.country}\"", f"CODE=\"{args.code}\"", f"VPN=\"{vpn}\""])
 
    ## Analysis
    print("Running analysis scripts")
    #run_command(["python3", "analysis/cdf.py", domain, "true", agg_func])
    #run_command(["python3", "analysis/boxplot.py", domain, "true", agg_func])
    #run_command(["python3", "analysis/bars.py", domain, "true", "--aggregate"])
    #run_command(["python3", "analysis/ips.py", domain, "true", agg_func])    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline')
    parser.add_argument('country', type=str, help='Country name to read and write JSON files')
    parser.add_argument('vpn', type=str, help='Country name to read and write JSON files')
    parser.add_argument('code', type=str, help='Country code to read and write JSON files')
    parser.add_argument('semester', type=str, help='Country code to read and write JSON files')
    parser.add_argument('count', type=str, help='Country code to read and write JSON files')
    parser.add_argument('process_repeated', type=str, help='Country code to read and write JSON files')
    parser.add_argument('agg_func', type=str, help='Country code to read and write JSON files')
    args = parser.parse_args()

    base_latency, geo_point = get_base_latency()
    print("Found base latency | geo point:", base_latency, geo_point)

    domain_folder = os.path.join("results", args.code)
    os.makedirs(domain_folder, exist_ok=True)
    locality_output = os.path.join(domain_folder, "locality")
    os.makedirs(locality_output, exist_ok=True)
    vpn_output = os.path.join(locality_output, args.vpn)
    os.makedirs(vpn_output, exist_ok=True)
    latency_output = os.path.join(domain_folder, "latency")
    os.makedirs(latency_output, exist_ok=True)
    results_output = os.path.join(domain_folder, "results")
    os.makedirs(results_output, exist_ok=True)
    os.makedirs(vpn_output, exist_ok=True)
    run_scripts(args.code, base_latency, geo_point, True, args.agg_func, args.vpn)
    
    if not args.process_repeated == "process_repeated": exit()
    
    # Process top 5 excluded domains
    os.makedirs(os.path.join("results", args.code, "repeated"), exist_ok=True)
    data = loads(open(f"results/{args.code}/repeated_output.json").read())
    data = {k: v for k, v in sorted(data.items(), key=lambda item: len(item[1]), reverse=True)[:5]}
    for idx, domain in enumerate(data.keys()):
        print("\033[95mStarting domain", domain, "\033[0m")
        
        # Create a folder for the domain in the results directory
        domain_folder = os.path.join("results", domain)
        os.makedirs(domain_folder, exist_ok=True)

        # Write the value of the key to an output.json file in the created folder
        output_file_path = os.path.join(domain_folder, "output.json")
        with open(output_file_path, "w") as output_file:
            json.dump(data[domain], output_file)

        run_scripts(domain, base_latency, geo_point, False, args.agg_func)
        run_command(["mv", f"results/{domain}", f"results/{args.code}/repeated/{domain}"])