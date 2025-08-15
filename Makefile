# =====================
# Global Variables
# =====================

PYTHON := python3
PYTHONPATH := $(PWD)
NODE := node

# =====================
# Pipeline general Script - We should use these scripts to run the pipeline
# =====================

setup:
	@echo "Setting up the environment..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/setup.py \
		--country "$(COUNTRY)" \
		--code "$(CODE)" \
		--semester "$(SEMESTER)"

latency:
	@echo "Measuring latency..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/latency.py \
		--country "$(COUNTRY)" \
		--code "$(CODE)"

locality:
	@echo "Measuring locality..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/locality.py \
		--country "$(COUNTRY)" \
		--code "$(CODE)" \
		--vpn "$(VPN)"

analysis:
	@echo "Running analysis..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/analysis.py \
		--country "$(COUNTRY)" \
		--code "$(CODE)"
 

# =====================
# Pipeline Steps Scripts
# =====================

fetch_websites:
	@echo "Fetching websites for $(COUNTRY)..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/steps/website_fetching/fetch_websites.py \
		--country "$(COUNTRY)" \
		--code "$(CODE)" \
		--query "$(QUERY)" \
		--semester "$(SEMESTER)" \
		--amount $(AMOUNT) \
		$(FILTER_DNS)

classify_websites:
	@echo "Classifying websites for $(COUNTRY)..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/steps/website_classification/classify_websites.py \
		--country "$(COUNTRY)" \
		--code "$(CODE)"

ping:
	@echo "Pinging websites for $(COUNTRY)..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/steps/latency/ping.py \
		--country "$(COUNTRY)" \
		--code "$(CODE)" \
		$(FETCH_FRESH_PROBES)

findcdn:
	@echo "Searching for websites CDNs for $(COUNTRY)..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/steps/locality/findcdn/findcdn.py \
		--country "$(COUNTRY)" \
		--code "$(CODE)" \
		--vpn "$(VPN)"

geolocate:
	@echo "Searching for websites CDNs locality  with ipinfo for $(COUNTRY)..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/steps/locality/geolocate/geolocate.py \
		--country "$(COUNTRY)" \
		--code "$(CODE)" \
		--vpn "$(VPN)"

locedge:
	@echo "Searching for websites CDNs locality with locedge for $(COUNTRY)..."
	@$(NODE) src/steps/locality/locedge/locedge.js \
		--country "$(COUNTRY)" \
		--code "$(CODE)" \
		--vpn "$(VPN)"



# =====================
# Analysis scripts
# =====================

analysis/locality/country:
	@echo "Running analysis of country dependency for $(COUNTRY)..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/steps/analysis/locality/bars_by_country.py \
		--country "$(COUNTRY)" \
		--code "$(CODE)" \
		$(if $(SAVE),--save) \
		$(if $(VPN),--vpn $(VPN)) \
		$(if $(ACCUMULATED),--accumulated)


analysis/locality/cdn_provider:
	@echo "Running analysis of cdn_provider dependency for $(COUNTRY)..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/steps/analysis/locality/bars_by_cdn.py \
		--country "$(COUNTRY)" \
		--code "$(CODE)" \
		$(if $(SAVE),--save) \
		$(if $(VPN),--vpn "$(VPN)") \
		$(if $(ACCUMULATED),--accumulated)


analysis/latency/cdf:
	@echo "Running analysis of latency cdf for $(COUNTRY)..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/steps/analysis/latency/cdf.py \
		--country "$(COUNTRY)" \
		--code "$(CODE)" \
		$(if $(SAVE),--save)

analysis/latency/difference:
	@echo "Running analysis of latency diffs for $(COUNTRY)..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/steps/analysis/latency/difference.py \
		--country "$(COUNTRY)" \
		--code "$(CODE)" \
		$(if $(SAVE),--save)

analysis/latency/prob_distribution:
	@echo "Running analysis of latency prob distribution for $(COUNTRY)..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/steps/analysis/latency/prob_distribution.py \
		--country "$(COUNTRY)" \
		--code "$(CODE)" \
		$(if $(SAVE),--save)
