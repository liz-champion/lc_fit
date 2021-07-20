GW170817_START = 1187008882.43
TMAX = 30.0
DATE=$(shell date +%Y%m%d)

RUN_LABEL=${DATE}${LABEL}

directories:
	mkdir -p pe_runs/

GW170817:
	#
	# Make the necessary directories
	#
	mkdir -p pe_runs/$@_${RUN_LABEL}/
	mkdir -p pe_runs/$@_${RUN_LABEL}/interp_files/
	mkdir -p pe_runs/$@_${RUN_LABEL}/condor_logs/
	#
	# Generate the initial parameter grid
	#
	python3 ${LC_FIT_LOCATION}/bin/generate_initial_grid.py --output-file pe_runs/$@_${RUN_LABEL}/grid_0.dat
	#
	# Parse the kilonova light curve data into our internal JSON format
	#
	python3 ${LC_FIT_LOCATION}/bin/parse_kn_data.py --t0 ${GW170817_START} --tmax ${TMAX} --json-file ${LC_FIT_LOCATION}/data/GW170817.json --bands g r i z y J H K --output-file pe_runs/$@_${RUN_LABEL}/lc_data.json
	#
	# Set up the DAG and submit files for the PE run
	#
	python3 ${LC_FIT_LOCATION}/bin/dag_setup.py --working-directory ${LC_FIT_LOCATION}/pe_runs/$@_${RUN_LABEL}/ --evaluate-interpolator-exe ${LC_FIT_LOCATION}/bin/evaluate_interpolator.py --partition-grid-exe ${LC_FIT_LOCATION}/bin/partition_grid.py --compute-posterior-exe ${LC_FIT_LOCATION}/bin/compute_posterior.py --generate-next-grid-exe ${LC_FIT_LOCATION}/bin/generate_next_grid.py --distance 40.0 --lc-file ${LC_FIT_LOCATION}/pe_runs/$@_${RUN_LABEL}/lc_data.json --bands g r i z y J H K --tempering-exponent-start 0.005

#
# Injection parameters
#
MEJ_DYN=0.05
MEJ_WIND=0.01
VEJ_DYN=0.2
VEJ_WIND=0.08
THETA=40.0
DISTANCE=40.0

TMIN=0.25
TMAX=25.0
NPTS=20
ERR=0.2

test:
	#
	# Make the necessary directories
	#
	mkdir -p pe_runs/$@_${RUN_LABEL}/
	mkdir -p pe_runs/$@_${RUN_LABEL}/interp_files/
	mkdir -p pe_runs/$@_${RUN_LABEL}/condor_logs/
	#
	# Generate a "grid" with just a single point to represent the true parameter values
	#
	python3 ${LC_FIT_LOCATION}/bin/generate_initial_grid.py --output-file ${LC_FIT_LOCATION}/pe_runs/$@_${RUN_LABEL}/injection_grid.dat --fixed-parameter mej_dyn ${MEJ_DYN} --fixed-parameter mej_wind ${MEJ_WIND} --fixed-parameter vej_dyn ${VEJ_DYN} --fixed-parameter vej_wind ${VEJ_WIND} --fixed-parameter theta ${THETA} --npts 1
	#
	# Make a DAG for generating the synthetic data
	#
	python3 ${LC_FIT_LOCATION}/bin/injection_dag_setup.py --working-directory ${LC_FIT_LOCATION}/pe_runs/$@_${RUN_LABEL}/ --evaluate-interpolator-exe ${LC_FIT_LOCATION}/bin/evaluate_interpolator.py --generate-lc-exe ${LC_FIT_LOCATION}/bin/generate_lc.py --injection-grid ${LC_FIT_LOCATION}/pe_runs/$@_${RUN_LABEL}/injection_grid.dat --distance ${DISTANCE} --tmin ${TMIN} --tmax ${TMAX} --output-file ${LC_FIT_LOCATION}/pe_runs/$@_${RUN_LABEL}/lc_data.json --npts-per-band ${NPTS} --error ${ERR} --theta ${THETA} --bands g r i z y J H K
	#
	# Generate the initial parameter grid
	#
	python3 ${LC_FIT_LOCATION}/bin/generate_initial_grid.py --output-file pe_runs/$@_${RUN_LABEL}/grid_0.dat
	#
	# Set up the DAG and submit files for the PE run
	#
	python3 ${LC_FIT_LOCATION}/bin/dag_setup.py --working-directory ${LC_FIT_LOCATION}/pe_runs/$@_${RUN_LABEL}/ --evaluate-interpolator-exe ${LC_FIT_LOCATION}/bin/evaluate_interpolator.py --partition-grid-exe ${LC_FIT_LOCATION}/bin/partition_grid.py --compute-posterior-exe ${LC_FIT_LOCATION}/bin/compute_posterior.py --generate-next-grid-exe ${LC_FIT_LOCATION}/bin/generate_next_grid.py --distance ${DISTANCE} --lc-file ${LC_FIT_LOCATION}/pe_runs/$@_${RUN_LABEL}/lc_data.json --bands g r i z y J H K --tempering-exponent-start 0.0005
