GW170817_START = 1187008882.43
TMAX = 30.0
DATE=$(shell date +%Y%m%d)

directories:
	mkdir -p pe_runs/

GW170817:
	mkdir -p pe_runs/$@_${DATE}/
	mkdir -p pe_runs/$@_${DATE}/interp_files/
	mkdir -p pe_runs/$@_${DATE}/condor_logs/
	python3 ${LC_FIT_LOCATION}/bin/generate_initial_grid.py --output-file pe_runs/$@_${DATE}/grid_0.dat
	python3 ${LC_FIT_LOCATION}/bin/parse_kn_data.py --t0 ${GW170817_START} --tmax ${TMAX} --json-file ${LC_FIT_LOCATION}/data/GW170817.json --bands g r i z y J H K --output-file pe_runs/$@_${DATE}/lc_data.json
	python3 ${LC_FIT_LOCATION}/bin/dag_setup.py --working-dir ${LC_FIT_LOCATION}/pe_runs/$@_${DATE}/ --evaluate-interpolator-exe ${LC_FIT_LOCATION}/bin/evaluate_interpolator.py --partition-grid-exe ${LC_FIT_LOCATION}/bin/partition_grid.py --compute-posterior-exe ${LC_FIT_LOCATION}/bin/compute_posterior.py --generate-next-grid-exe ${LC_FIT_LOCATION}/bin/generate_next_grid.py --distance 40.0 --lc-file ${LC_FIT_LOCATION}/pe_runs/$@_${DATE}/lc_data.json --bands g r i z y J H K --tempering-exponent-start 0.005
