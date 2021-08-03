GW170817_START = 1187008882.43
TMAX = 30.0
DATE=$(shell date +%Y%m%d)

RUN_LABEL=${DATE}${LABEL}

NPTS_PER_ITERATION=25000

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
	python3 ${LC_FIT_LOCATION}/bin/generate_initial_grid.py --output-file pe_runs/$@_${RUN_LABEL}/grid_0.dat --npts ${NPTS_PER_ITERATION} --gaussian-prior theta 20.0 5.0
	#
	# Parse the kilonova light curve data into our internal JSON format
	#
	python3 ${LC_FIT_LOCATION}/bin/parse_kn_data.py --t0 ${GW170817_START} --tmax ${TMAX} --json-file ${LC_FIT_LOCATION}/data/GW170817.json --bands g r i z y J H K --output-file pe_runs/$@_${RUN_LABEL}/lc_data.json
	#
	# Set up the DAG and submit files for the PE run
	#
	python3 ${LC_FIT_LOCATION}/bin/dag_setup.py --working-directory ${LC_FIT_LOCATION}/pe_runs/$@_${RUN_LABEL}/ --evaluate-interpolator-exe ${LC_FIT_LOCATION}/bin/evaluate_interpolator.py --partition-grid-exe ${LC_FIT_LOCATION}/bin/partition_grid.py --compute-posterior-exe ${LC_FIT_LOCATION}/bin/compute_posterior.py --generate-next-grid-exe ${LC_FIT_LOCATION}/bin/generate_next_grid.py --distance 40.0 --lc-file ${LC_FIT_LOCATION}/pe_runs/$@_${RUN_LABEL}/lc_data.json --bands g r i z y J H K --tempering-exponent-start 0.001 --tempering-exponent-iterations 8 --npts-per-iteration ${NPTS_PER_ITERATION} --generate-next-grid-args "--gaussian-prior theta 20.0 5.0"

#
# Injection parameters
#
MEJ_DYN=0.05
MEJ_WIND=0.01
VEJ_DYN=0.2
VEJ_WIND=0.15
THETA=20.0
DISTANCE=40.0

TMIN=0.25
TMAX=25.0
NPTS=20
ERR=0.25

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
	python3 ${LC_FIT_LOCATION}/bin/generate_initial_grid.py --output-file pe_runs/$@_${RUN_LABEL}/grid_0.dat --npts ${NPTS_PER_ITERATION}
	#
	# Set up the DAG and submit files for the PE run
	#
	python3 ${LC_FIT_LOCATION}/bin/dag_setup.py --working-directory ${LC_FIT_LOCATION}/pe_runs/$@_${RUN_LABEL}/ --evaluate-interpolator-exe ${LC_FIT_LOCATION}/bin/evaluate_interpolator.py --partition-grid-exe ${LC_FIT_LOCATION}/bin/partition_grid.py --compute-posterior-exe ${LC_FIT_LOCATION}/bin/compute_posterior.py --generate-next-grid-exe ${LC_FIT_LOCATION}/bin/generate_next_grid.py --distance ${DISTANCE} --lc-file ${LC_FIT_LOCATION}/pe_runs/$@_${RUN_LABEL}/lc_data.json --bands g r i z y J H K --tempering-exponent-start 0.01 --npts-per-iteration ${NPTS_PER_ITERATION} --tempering-exponent-iterations 5

#
# An injection/recovery test with simulation data
#

# Take the values from the simulation data
MEJ_DYN=0.097050
MEJ_WIND=0.083748
VEJ_DYN=0.197642
VEJ_WIND=0.297978

# Choose an angular bin to pull from in the simulation data, and calculate the corresponding angle
ANGULAR_BIN=24
THETA=$(shell python3 ${LC_FIT_LOCATION}/bin/convert_angular_bin_to_theta.py --angular-bin ${ANGULAR_BIN})

# Luminosity distance
DISTANCE=40.0

# Other injection parmeters/arguments
TMIN=0.25
TMAX=25.0
NPTS=20
ERR=0.25

simulation:
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
	# Extract the light curves from the simulation files
	#
	python3 ${LC_FIT_LOCATION}/bin/parse_sim_kn_data.py --simulation-file ${LC_FIT_LOCATION}/data/Run_TP_dyn_all_lanth_wind2_all_md0.097050_vd0.197642_mw0.083748_vw0.297978_mags_2020-11-03.dat --tmin ${TMIN} --tmax ${TMAX} --npts-per-band ${NPTS} --error ${ERR} --output-file ${LC_FIT_LOCATION}/pe_runs/$@_${RUN_LABEL}/lc_data.json --angular-bin ${ANGULAR_BIN} --distance ${DISTANCE} --bands g r i z y J H K
	#
	# Generate the initial parameter grid
	#
	python3 ${LC_FIT_LOCATION}/bin/generate_initial_grid.py --output-file pe_runs/$@_${RUN_LABEL}/grid_0.dat --npts ${NPTS_PER_ITERATION}
	#
	# Set up the DAG and submit files for the PE run
	#
	python3 ${LC_FIT_LOCATION}/bin/dag_setup.py --working-directory ${LC_FIT_LOCATION}/pe_runs/$@_${RUN_LABEL}/ --evaluate-interpolator-exe ${LC_FIT_LOCATION}/bin/evaluate_interpolator.py --partition-grid-exe ${LC_FIT_LOCATION}/bin/partition_grid.py --compute-posterior-exe ${LC_FIT_LOCATION}/bin/compute_posterior.py --generate-next-grid-exe ${LC_FIT_LOCATION}/bin/generate_next_grid.py --distance ${DISTANCE} --lc-file ${LC_FIT_LOCATION}/pe_runs/$@_${RUN_LABEL}/lc_data.json --bands g r i z y J H K --tempering-exponent-start 0.001 --npts-per-iteration ${NPTS_PER_ITERATION} --tempering-exponent-iterations 15
