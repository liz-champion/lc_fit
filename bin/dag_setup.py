#! /usr/bin/env python3

#
# Generates Condor .sub and .dag files for the iterative workflow.
# A lot of this is adapted from Vera's code.
#

import numpy as np
import json
import os
import argparse

#
# Objects representing individual condor jobs and the DAG (note: DAG = "Directed Acyclic Graph")
#

# NOTE: If you change this function, make sure you change it in injection_dag_setup.py too.
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
        requirements = ["TARGET.GLIDEIN_ResourceName =!= MY.MachineAttrGLIDEIN_ResourceName{0}".format(i + 1) for i in range(retries)]
        if len(requirements) > 0:
            fp.write("Requirements = {0}\n".format(" && \ \n".join(requirements)))

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

parser = argparse.ArgumentParser(description="Generates Condor .sub and .dag files for the iterative workflow")
parser.add_argument("--working-directory", help="Working directory for the PE run")
parser.add_argument("--evaluate-interpolator-exe", help="Location of `evaluate_interpolator.py`")
parser.add_argument("--partition-grid-exe", help="Location of `partition_grid.py`")
parser.add_argument("--compute-posterior-exe", help="Location of `compute_posterior.py`")
parser.add_argument("--compute-lnL-exe", help="Location of `compute_lnL.py`")
parser.add_argument("--generate-next-grid-exe", help="Location of `generate_next_grid.py`")
parser.add_argument("--lc-file", help="Location of JSON file created by `parse_kn_data.py`")
parser.add_argument("--n-iterations", type=int, default=10, help="Number of sampling iterations")
parser.add_argument("--bands", nargs="+", help="Bands to use for PE (must be present in the light curve JSON file)")
parser.add_argument("--distance", type=float, help="Luminosity distance (in Mpc)")
parser.add_argument("--tempering-exponent-start", type=float, default=1., help="Starting value for tempering exponent")
parser.add_argument("--tempering-exponent-iterations", type=float, default=5, help="Number of iterations over which to increase the tempering exponent to 1")
parser.add_argument("--npts-per-iteration", type=int, default=25000, help="Number of points to sample per iteration")
args = parser.parse_args()

#
# Generate our .sub files
#

RETRIES = 4 # For now just hard-code a number of retries

# partition_grid.sub
tag = "$(iteration)"
generate_submit_file("partition_grid",
        args.partition_grid_exe,
        "--grid-file {0}grid_$(iteration).dat --output-directory {0}interp_files/$(iteration)/".format(args.working_directory),
        args.working_directory,
        tag=tag,
        memory=2,
        disk=2,
        retries=RETRIES)

# eval_interp.sub
tag = "$(interp_time)_$(interp_angle)_$(band)"
generate_submit_file("eval_interp",
        args.evaluate_interpolator_exe,
        "--interp-angle $(interp_angle) --interp-time $(interp_time) --band $(band) --grid-file {0}grid_$(iteration).dat --index-file {0}interp_files/$(iteration)/indices_$(interp_angle).dat --output-directory {0}interp_files/$(iteration)/".format(args.working_directory),
        args.working_directory,
        tag=tag,
        memory=16,
        disk=4,
        retries=RETRIES)

# compute_posterior.sub.
tag = "$(iteration)"
generate_submit_file("compute_posterior",
        args.compute_posterior_exe,
        "--interp-directory {0}interp_files/$(iteration)/ --output-file {0}posterior-samples_$(iteration).dat --grid-file {0}grid_$(iteration).dat --lc-file {1} --bands {2} --distance {3}".format(args.working_directory, args.lc_file, " ".join(args.bands), args.distance),
        args.working_directory,
        tag=tag,
        memory=4,
        disk=2,
        retries=RETRIES)

# generate_next_grid.sub
tag = "$(iteration)"
generate_submit_file("generate_next_grid",
        args.generate_next_grid_exe,
        "--posterior-file {0}posterior-samples_$(iteration).dat --output-file {0}grid_$(next_iteration).dat --tempering-exponent $(exponent) --npts {1}".format(args.working_directory, args.npts_per_iteration),
        args.working_directory,
        tag=tag,
        memory=4,
        cpus=2,
        disk=2,
        retries=RETRIES)

#
# The rest of this script generates the DAG file.
# First we need to determine which interpolators might need to be used.
#

# Load the light curve data
with open(args.lc_file, "r") as fp:
    lc_data = json.load(fp)

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
        for angle in ["0", "30", "45", "60", "75", "90"]:
            # This might make it look as if we're actually evaluating every angle at a given time.
            # This is actually not the case, because the interpolator evaluation script will immediately exit if its parameter file is empty,
            # which happens when there are no samples in the grid between two angles.
            interpolator_args_set.add((t_before, angle, band))
            interpolator_args_set.add((t_after, angle, band))

# Precompute the tempering exponents
tempering_exponents = np.ones(args.n_iterations)
tempering_exponents[:args.tempering_exponent_iterations] = np.logspace(np.log10(args.tempering_exponent_start), 0., args.tempering_exponent_iterations)

dag_fname = args.working_directory + "run.dag"
child_parent_list = []
with open(dag_fname, "w") as fp:
    for iteration in range(args.n_iterations):
        #
        # DAG node for partitioning the grid by viewing angle
        #
        fp.write("JOB partition_grid_{0} partition_grid.sub\n".format(iteration))
        fp.write("VARS partition_grid_{0} iteration=\"{0}\"\n".format(iteration))
        fp.write("RETRY partition_grid_{0} {1}\n".format(iteration, RETRIES))
        if iteration > 0:
            fp.write("PARENT generate_next_grid_{0} CHILD partition_grid_{1}\n".format(iteration - 1, iteration))
        #
        # DAG nodes to evaluate every interpolator that is needed
        #
        interp_job_list = []
        for (interp_time, interp_angle, band) in interpolator_args_set:
            tag = "{0}_{1}_{2}".format(interp_time, interp_angle, band)
            fp.write("JOB eval_interp_{0}_{1} eval_interp.sub\n".format(tag, iteration))
            fp.write("VARS eval_interp_{0}_{1} interp_time=\"{2}\" interp_angle=\"{3}\" band=\"{4}\" iteration=\"{1}\"\n".format(tag, iteration, interp_time, interp_angle, band))
            fp.write("RETRY eval_interp_{0}_{1} {2}\n".format(tag, iteration, RETRIES))
            interp_job_list.append("eval_interp_{0}_{1}".format(tag, iteration))
        for job in interp_job_list:
            child_parent_list.append("PARENT partition_grid_{0} CHILD {1}".format(iteration, job))
        #
        # DAG node to compute the posterior
        #
        fp.write("JOB compute_posterior_{0} compute_posterior.sub\n".format(iteration))
        fp.write("VARS compute_posterior_{0} iteration=\"{0}\"\n".format(iteration))
        fp.write("RETRY compute_posterior_{0} {1}\n".format(iteration, RETRIES))
        for job in interp_job_list:
            child_parent_list.append("PARENT {0} CHILD compute_posterior_{1}".format(job, iteration))
        #
        # DAG node to generate the next grid
        #
        fp.write("JOB generate_next_grid_{0} generate_next_grid.sub\n".format(iteration))
        fp.write("VARS generate_next_grid_{0} iteration=\"{0}\" next_iteration=\"{1}\" exponent=\"{2}\"\n".format(iteration, iteration + 1, tempering_exponents[iteration]))
        fp.write("PARENT compute_posterior_{0} CHILD generate_next_grid_{0}\n".format(iteration))
        fp.write("RETRY generate_next_grid_{0} {1}\n".format(iteration, RETRIES))

    fp.write("\n".join(child_parent_list))
