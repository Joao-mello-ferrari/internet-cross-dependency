import csv
import re
import ipaddress

def ip_to_int(ip):
    """Convert IP string to integer."""
    return int(ipaddress.IPv4Address(ip))

# Define the input and output file paths
INPUT_CSV = "csv.csv"  # Replace with your actual input file
OUTPUT_CSV = "csv_ipv4.csv"

# Define the regex pattern for a valid IPv4 address
ipv4_regex = re.compile(r'^\d+\.\d+\.\d+\.\d+$')

# Open input file in read mode and output file in write mode
with open(INPUT_CSV, mode='r', newline='', encoding='utf-8') as infile, \
     open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Read and write the header row
    header = next(reader)
    writer.writerow(header + ["start_ip_int", "end_ip_int"])
    
    # Process each row
    c = 0
    for row in reader:
        c += 1
        if c % 1000 == 0:
          print("processing row", c)
          
        if len(row) < 2:
            continue  # Skip malformed rows
        
        start_ip, end_ip = row[0], row[1]  # Last two columns
        
        # Check if both IPs match the IPv4 regex
        if ipv4_regex.match(start_ip) and ipv4_regex.match(end_ip):
            writer.writerow(row + [ip_to_int(start_ip), ip_to_int(end_ip)])  # Write valid rows to the output file


print("✅ Filtering complete! Results saved to", OUTPUT_CSV)




