import json

with open("input.csv", "r") as f:
    lines = f.readlines()#.split('\n')

lines = lines[1:]

out = {}

for line in lines:
    parts = line.strip().split(',')
    name, _as, v4, v6, percent = parts

    out[name.lower()] = {
        "name": name,
        "ases": {},
        "percent": 0
    }

for line in lines:
    parts = line.strip().split(',')
    name, _as, v4, v6, percent = parts

    if _as.strip() == '':
        continue

    v4 = v4.replace(' ', '').split(';') if v4.replace(' ', '').split(';')[0] != '' else []
    v6 = v6.replace(' ', '').split(';') if v6.replace(' ', '').split(';')[0] != '' else []
    out[name.lower()]["ases"][_as.strip()] = {
        "v4": v4,
        "v6": v6,
        "percent": float(percent.strip())
    }

    out[name.lower()]["percent"] += float(percent.strip())

print(out)

with open("probes.json", "r") as f:
    probes = json.load(f)

p = [] 
country_ases = probes.get("br", {}).get("ases", {})
for country_as in country_ases.values():
  p.extend(country_as.get("v4", []))

print(p)