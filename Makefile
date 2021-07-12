GW170817_START = 1187008882.43
TMAX = 30.0
DATE=$(shell date +%Y%m%d)

directories:
	mkdir -p pe_runs/

GW170817:
	mkdir -p pe_runs/$@_${DATE}/
	python3 ${LC_FIT_LOCATION}/bin/parse_kn_data.py --t0 ${GW170817_START} --tmax ${TMAX} --json-file ${LC_FIT_LOCATION}/data/GW170817.json --bands g r i z y J H K --output-file pe_runs/$@_${DATE}/lc_data.json
