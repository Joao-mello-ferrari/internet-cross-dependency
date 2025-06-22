# =====================
# Global Variables
# =====================

PYTHON := python3
PYTHONPATH := $(PWD)
NODE := node

fetch_websites:
	@echo "Fetching websites for $(COUNTRY)..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) steps/website_fetching/fetch_websites.py \
		--country "$(COUNTRY)" \
		--code "$(CODE)" \
		--query "$(QUERY)" \
		--semester "$(SEMESTER)" \
		--amount $(AMOUNT) \
		$(FILTER_DNS)

ping:
	@echo "Pinging websites for $(COUNTRY)..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) steps/latency/ping.py \
		--country "$(COUNTRY)" \
		--code "$(CODE)" \
		$(FETCH_FRESH_PROBES)

findcdn:
	@echo "Searching for websites CDNs for $(COUNTRY)..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) steps/locality/findcdn/findcdn.py \
		--country "$(COUNTRY)" \
		--code "$(CODE)" \
		--vpn "$(VPN)"

geolocate:
	@echo "Searching for websites CDNs locality  with ipinfo for $(COUNTRY)..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) steps/locality/geolocate/geolocate.py \
		--country "$(COUNTRY)" \
		--code "$(CODE)" \
		--vpn "$(VPN)"

locedge:
	@echo "Searching for websites CDNs locality with locedge for $(COUNTRY)..."
	@$(NODE) steps/locality/locedge/locedge.js \
		--country "$(COUNTRY)" \
		--code "$(CODE)" \
		--vpn "$(VPN)"

latency:
	@echo "Running latency measurements..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) steps/latency/run_latency.py --country "ca"

