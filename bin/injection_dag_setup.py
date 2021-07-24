#! /usr/bin/env python3

#
# Generates Condor .sub and .dag files for the iterative workflow.
# A lot of this is adapted from Vera's code.
#

import numpy as np
import json
import os
import argparse

# NOTE: If you change this function, make sure you change it in dag_setup.py too.
# I know I should import this function from one place, but I wanted all these scripts to be completely independent.
def generate_submit_file(
        sub_fname, # the name of the submit file to create
        exe, # the executable for the submit file
        arguments,
        working_directory, # where to save submit file
        tag="", # how to label the log, out, err, and output files (can use variables defined in DAG)
        input_files=[],
        output_files=[],
        cpus=1,
        memory=1,
        disk=1,
        retries=0):

    with open(working_directory + sub_fname + ".sub", "w") as fp:
        # Write signature comment in file
        fp.write("# Autmatically generated job submit file\n")

        # Write the project name for submission
        fp.write("\n# Project name\n")
        fp.write("+ProjectName = \"lc_fit\"\n")

        # Default universe
        fp.write("universe = vanilla\n")

        # Don't retry on the same node you failed on
        # Thanks Richard!
        #requirements = ["TARGET.GLIDEIN_ResourceName =!= MY.MachineAttrGLIDEIN_ResourceName{0}".format(i + 1) for i in range(retries)]
        #if len(requirements) > 0:
        #    fp.write("Requirements = {0}\n".format(" && \ \n".join(requirements)))

        # Write the location of the executable
        fp.write("\n# Location of the executable\n")
        fp.write("executable = {0}\n".format(exe))

        # Write log file locations
        fp.write("\n# Log files:\n")
        fp.write("error = {0}condor_logs/{1}_{2}.err\n".format(working_directory, sub_fname, tag))
        fp.write("output = {0}condor_logs/{1}_{2}.out\n".format(working_directory, sub_fname, tag))
        fp.write("log = {0}condor_logs/{1}_{2}.log\n".format(working_directory, sub_fname, tag))

        # Make sure you transfer output files at the right time
        fp.write("\nShouldTransferFiles = True\n")
        fp.write("WhenToTransferOutput = ON_EXIT\n")

        fp.write("\ngetenv = True\n")

        fp.write("\ninitialdir = {0}\n".format(working_directory))
        
        # Arguments
        fp.write("\n# Arguments:\n")
        fp.write("arguments = {0}\n".format(arguments))

        # Request Resources
        fp.write("\n# Resources:\n")
        fp.write("request_cpus = {0}\n".format(cpus))
        fp.write("request_memory = {0} MB\n".format(memory * 1024))
        fp.write("request_disk = {0} MB\n".format(disk * 1024))

        user = os.environ["USER"]
        fp.write("\naccounting_group = ligo.dev.o3.cbc.pe.lalinferencerapid\n")
        fp.write("accounting_group_user = {0}\n".format(user))

        fp.write("\n# Number of identical jobs to run: \n")
        fp.write("queue 1\n")

parser = argparse.ArgumentParser(description="Makes a DAG file and associated submit files for generating injection data")
parser.add_argument("--working-directory", help="Working directory for the PE run")
parser.add_argument("--evaluate-interpolator-exe", help="Location of `evaluate_interpolator.py`")
parser.add_argument("--generate-lc-exe", help="Location of `generate_lc.py`")
parser.add_argument("--injection-grid", help="File containing injection parameters as a single entry in a 'grid'")
parser.add_argument("--distance", type=float, help="Luminosity distance (in Mpc)")
parser.add_argument("--theta", type=float, help="Viewing angle (in degrees)")
parser.add_argument("--tmin", type=float, help="Minimum time value")
parser.add_argument("--tmax", type=float, help="Maximum time value")
parser.add_argument("--output-file", help="Filename to save the final JSON light curve data to")
parser.add_argument("--npts-per-band", type=int, help="Number of data points per band")
parser.add_argument("--bands", nargs="+", help="Bands to generate data for")
parser.add_argument("--error", type=float, default=0.2, help="Standard deviation of errors (in magnitude) for synthetic data")
args = parser.parse_args()

RETRIES = 4 # For now just hard-code a number of retries

# eval_interp_injection.sub
tag = "$(interp_time)_$(interp_angle)_$(band)"
generate_submit_file("eval_interp_injection",
        args.evaluate_interpolator_exe,
        "--interp-angle $(interp_angle) --interp-time $(interp_time) --band $(band) --grid-file {1} --output-directory {0}interp_files/injection/".format(args.working_directory, args.injection_grid),
        args.working_directory,
        tag=tag,
        memory=4,
        disk=2,
        retries=RETRIES)

# generate_lc_injection.sub
generate_submit_file("generate_lc_injection",
        args.generate_lc_exe,
        "--interp-directory {0}interp_files/injection/ --lc-file {1} --bands {2} --distance {3} --theta {4} --error {5}".format(args.working_directory, args.output_file, " ".join(args.bands), args.distance, args.theta, args.error),
        args.working_directory,
        memory=4,
        disk=2,
        retries=RETRIES)

# Initialize a dictionary (which will eventually become a JSON file) for our synthetic data
lc_data = {band:{} for band in args.bands}
for band in args.bands:
    # Generate random time values that are uniformly spaced on a log scale, and make sure they're sorted
    lc_data[band]["time"] = list(np.sort(np.exp(np.random.uniform(np.log(args.tmin), np.log(args.tmax), args.npts_per_band))))
    lc_data[band]["mag"] = []
    lc_data[band]["mag_err"] = [args.error] * args.npts_per_band

# Which interpolators are needed depends on the times at which we have magnitude data.
# We want to use the interpolators immediately before and after each data point in a particular band, and not waste resources evaluating the rest.
# Note that the following code assumes there is always an interpolator before and after each data point.
# Basically that means make sure the time value of every data point used for PE is 0.125 < t < 37.2.
interpolator_times = np.logspace(np.log10(0.125), np.log10(37.239195485411194), 264)
interpolator_args_set = set() # use a set so we don't end up with multiple copies of the same interpolator
for band in args.bands:
    for t_data in lc_data[band]["time"]:
        for i, t_interp in enumerate(interpolator_times):
            if t_interp > t_data:
                break
        t_before = "{:.3f}".format(interpolator_times[i - 1])
        t_after = "{:.3f}".format(interpolator_times[i])
        angles = ["0", "30", "45", "60", "75", "90"]
        for i in range(1, len(angles)):
            if float(angles[i]) > args.theta:
                interpolator_args_set.add((t_before, angles[i - 1], band))
                interpolator_args_set.add((t_after, angles[i], band))
                break

dag_fname = args.working_directory + "generate_data.dag"
child_parent_list = []
with open(dag_fname, "w") as fp:
    interp_job_list = []
    for (interp_time, interp_angle, band) in interpolator_args_set:
        tag = "{0}_{1}_{2}".format(interp_time, interp_angle, band)
        fp.write("JOB eval_interp_injection_{0} eval_interp_injection.sub\n".format(tag))
        fp.write("VARS eval_interp_injection_{0} interp_time=\"{1}\" interp_angle=\"{2}\" band=\"{3}\"\n".format(tag, interp_time, interp_angle, band))
        fp.write("RETRY eval_interp_injection_{0} {1}\n".format(tag, RETRIES))
        interp_job_list.append("eval_interp_injection_{0}".format(tag))
    fp.write("JOB generate_lc_injection generate_lc_injection.sub\n")
    fp.write("RETRY generate_lc_injection {0}\n".format(RETRIES))
    for job in interp_job_list:
        fp.write("PARENT {0} CHILD generate_lc_injection\n".format(job))

with open(args.output_file, "w") as fp:
    json.dump(lc_data, fp, indent=4)
