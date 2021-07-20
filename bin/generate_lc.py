#! /usr/bin/env python3

#
# Stitch together interpolator evaluations to generate injection/recovery data
#

import numpy as np
import argparse
import json
import os
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser(description="Creates a light curve JSON file for injection/recovery tests from interpolator evaluations")
parser.add_argument("--interp-directory", help="Directory containing the interpolator evaluations")
parser.add_argument("--lc-file", help="Name of light curve JSON file")
parser.add_argument("--bands", nargs="+", help="Bands to use")
parser.add_argument("--distance", type=float, help="Luminosity distance (in Mpc)")
parser.add_argument("--theta", type=float, help="Viewing angle (in degrees)")
parser.add_argument("--error", type=float, default=0.2, help="Standard deviation of errors (in magnitude) for synthetic data")
args = parser.parse_args()

# Load the light curve data (which for now only includes times)
with open(args.lc_file, "r") as fp:
    lc_data = json.load(fp)
for b in lc_data.keys():
    for key in lc_data[b].keys():
        lc_data[b][key] = np.array(lc_data[b][key]) # cast everything as a numpy array
 
interpolator_times = np.logspace(np.log10(0.125), np.log10(37.239195485411194), 264)
interpolator_times_str_to_float = {"{:.3f}".format(_t):_t for _t in interpolator_times} # this is useful since the times in the filenames are truncated to three decimal places

# Set up a data structure to hold all the interpolated magnitudes.
#   - A dictionary mapping (band, interpolator_angle) to *another* dictionary.
#       - This sub-dictionary maps the strings "time", "mag", and "mag_err" to lists containing those values
interp_data = {}

# Fill this data structure with data loaded from the interpolated magnitude files
fname_base = args.interp_directory + ("/" if args.interp_directory[-1] != "/" else "")
for fname in os.listdir(args.interp_directory):

    fname_split = fname.split("_")

    # Determine whether it's a file we want or not
    if fname_split[0] != "eval":
        continue

    fname_split = fname_split[2:]
    if fname_split[0] == "injection":
        fname_split = fname_split[1:]

    # Get the time, angle, and band from the filename
    t_interp = interpolator_times_str_to_float[fname_split[0]] # convert from the truncated time in the filename to the full time value
    theta_interp = float(fname_split[1])
    band = fname_split[2][0]

    if band not in args.bands:
        continue

    # Load the interpolated magnitude data
    data = np.loadtxt(fname_base + fname)
    
    if (band, theta_interp) not in interp_data.keys():
        interp_data[(band, theta_interp)] = {"time":[], "mag":[]}
    interp_data[(band, theta_interp)]["time"].append(t_interp)
    interp_data[(band, theta_interp)]["mag"].append(data[0])

# Go through and sort every array in interp_data by time (and also convert them to numpy arrays)
for key in interp_data.keys():
    interp_data[key]["time"] = np.array(interp_data[key]["time"])
    interp_data[key]["mag"] = np.array(interp_data[key]["mag"])
    sorted_indices = np.argsort(interp_data[key]["time"])
    interp_data[key]["time"] = interp_data[key]["time"][sorted_indices]
    interp_data[key]["mag"] = interp_data[key]["mag"][sorted_indices]

# Determine the angular bin
interp_angles = [0., 30., 45., 60., 75., 90.]
for j in range(1, len(interp_angles)):
    if interp_angles[j] > args.theta:
        break
theta_lower = interp_angles[j - 1]
theta_upper = interp_angles[j]

# Create interpolators for each band, where the interpolation over angle has been done
f_dict = {}
for b in args.bands:
    # Get the interpolated values for this row in the grid
    t = interp_data[(b, theta_lower)]["time"]
    mag_lower = interp_data[(b, theta_lower)]["mag"]
    mag_upper = interp_data[(b, theta_upper)]["mag"]
    
    # Interpolate in angle
    mag = ((theta_upper - args.theta) / (theta_upper - theta_lower)) * mag_lower + ((args.theta - theta_lower) / (theta_upper - theta_lower)) * mag_upper

    # Interpolate in time and put the interpolators in the dictionary
    f_dict[b] = interp1d(t, mag, fill_value="extrapolate")

# Now, finally, fill our JSON data structure
for b in args.bands:
    lc_data[b]["mag"] = f_dict[b](lc_data[b]["time"]) + 5. * (np.log10(args.distance * 1.e6) - 1.)
    # Add some Gaussian noise
    lc_data[b]["mag"] += np.random.normal(0., args.error, len(lc_data[b]["mag"]))

# Cast things as lists so they're JSON serializable
for b in args.bands:
    for key in lc_data[b].keys():
        lc_data[b][key] = list(lc_data[b][key])

# Write the data to the file
with open(args.lc_file, "w") as fp:
    json.dump(lc_data, fp, indent=4)
