#! /usr/bin/env python3

import numpy as np
import argparse
import json
import os
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser(description="Compute log-likelihoods from light curve evaluations")
parser.add_argument("--interp-directory", help="Directory containing the interpolator evaluations and index files")
parser.add_argument("--output-file", help="Name of file to save posterior samples to")
parser.add_argument("--grid-file", help="Location of the grid file")
parser.add_argument("--lc-file", help="Location of the light curve JSON file")
parser.add_argument("--bands", nargs="+", help="Bands to use")
parser.add_argument("--distance", type=float, help="Luminosity distance (in Mpc)")
args = parser.parse_args()

# Load the files containing the grid indices corresponding to each angle
indices_fname_base = args.interp_directory + ("/" if args.interp_directory[-1] != "/" else "") + "indices_"
interp_angles = [0., 30., 45., 60., 75., 90.]
indices = {_theta_interp:np.loadtxt(indices_fname_base + str(int(_theta_interp)) + ".dat").astype(int) for _theta_interp in interp_angles}

# Load the grid
grid = np.loadtxt(args.grid_file)
theta_grid = grid[:,-1] # angle is the fifth column

# Load the light curve data
with open(args.lc_file, "r") as fp:
    lc_data = json.load(fp)
for b in lc_data.keys():
    for key in lc_data[b].keys():
        lc_data[b][key] = np.array(lc_data[b][key]) # cast everything as a numpy array

interpolator_times = np.logspace(np.log10(0.125), np.log10(37.239195485411194), 264)
interpolator_times_str_to_float = {"{:.3f}".format(_t):_t for _t in interpolator_times} # this is useful since the times in the filenames are truncated to three decimal places

# Set up a data structure to hold all the interpolated magnitudes.
# This is a list with the following structure:
#   - Each entry corresponds to a row in the grid.
#   - Each entry is a dictionary mapping (band, interpolator_angle) to *another* dictionary.
#       - This sub-dictionary maps the strings "time", "mag", and "mag_err" to lists containing those values
# The point is to get all the interpolated magnitudes in one place, since they currently exist in many different files,
# all while keeping track of which grid point each one corresponds to, its time, and its band, and to minimize lookup times going forward.
interp_data = [{} for _ in range(theta_grid.size)]

# Fill this data structure with data loaded from the interpolated magnitude files
fname_base = args.interp_directory + ("/" if args.interp_directory[-1] != "/" else "")
for fname in os.listdir(args.interp_directory):

    fname_split = fname.split("_")

    # Determine whether it's a file we want or not
    if fname_split[0] != "eval":
        continue
    
    fname_split = fname_split[2:]
    
    # Get the time, angle, and band from the filename
    t_interp = interpolator_times_str_to_float[fname_split[0]] # convert from the truncated time in the filename to the full time value
    theta_interp = float(fname_split[1])
    band = fname_split[2][0]

    if band not in args.bands:
        continue

    # Load the interpolated magnitude data
    data = np.loadtxt(fname_base + fname)
    
    # "i" is the index within this data file
    # "grid_index" is the row of the grid corresponding to this particular interpolator evaluation
    for i, grid_index in enumerate(indices[theta_interp]):
        # initialize things if needed
        if (band, theta_interp) not in interp_data[grid_index].keys():
            interp_data[grid_index][(band, theta_interp)] = {"time":[], "mag":[], "mag_err":[]}
        interp_data[grid_index][(band, theta_interp)]["time"].append(t_interp)
        interp_data[grid_index][(band, theta_interp)]["mag"].append(data[i][0])
        interp_data[grid_index][(band, theta_interp)]["mag_err"].append(data[i][1])

# Go through and sort every array in interp_data by time (and also convert them to numpy arrays)
for i in range(theta_grid.size):
    for key in interp_data[i].keys():
        interp_data[i][key]["time"] = np.array(interp_data[i][key]["time"])
        interp_data[i][key]["mag"] = np.array(interp_data[i][key]["mag"])
        interp_data[i][key]["mag_err"] = np.array(interp_data[i][key]["mag_err"])
        sorted_indices = np.argsort(interp_data[i][key]["time"])
        interp_data[i][key]["time"] = interp_data[i][key]["time"][sorted_indices]
        interp_data[i][key]["mag"] = interp_data[i][key]["mag"][sorted_indices]
        interp_data[i][key]["mag_err"] = interp_data[i][key]["mag_err"][sorted_indices]

# Now build a list of interpolators corresponding to each row in the grid for each band, where the interpolation over angle has been done
lc_functions = []
for i, theta in enumerate(theta_grid):
    # Determine the angular bin
    for j in range(1, len(interp_angles)):
        if interp_angles[j] > theta:
            break
    theta_lower = interp_angles[j - 1]
    theta_upper = interp_angles[j]

    f_dict = {}
    for b in args.bands:
        # Get the interpolated values for this row in the grid
        t = interp_data[i][(b, theta_lower)]["time"]
        mag_lower = interp_data[i][(b, theta_lower)]["mag"]
        mag_err_lower = interp_data[i][(b, theta_lower)]["mag_err"]
        mag_upper = interp_data[i][(b, theta_upper)]["mag"]
        mag_err_upper = interp_data[i][(b, theta_upper)]["mag_err"]
        
        # Interpolate in angle
        mag = ((theta_upper - theta) / (theta_upper - theta_lower)) * mag_lower + ((theta - theta_lower) / (theta_upper - theta_lower)) * mag_upper
        mag_err = ((theta_upper - theta) / (theta_upper - theta_lower)) * mag_err_lower + ((theta - theta_lower) / (theta_upper - theta_lower)) * mag_err_upper

        # Interpolate in time and put the interpolators in the dictionary
        f_dict[b] = (interp1d(t, mag, fill_value="extrapolate"), interp1d(t, mag_err, fill_value="extrapolate"))

    lc_functions.append(f_dict)

# Now, finally, compute the likelihoods
lnL = np.zeros(theta_grid.size)
for i, f_dict in enumerate(lc_functions):
    for b in args.bands:
        mag_interpolator, mag_err_interpolator = f_dict[b]
        # make sure to account for the distance modulus when computing residuals
        residuals = mag_interpolator(lc_data[b]["time"]) + (5. * (np.log10(args.distance * 1.e6) - 1.)) - lc_data[b]["mag"]
        model_error = mag_err_interpolator(lc_data[b]["time"])
        lnL[i] += -0.5 * np.sum(residuals**2 / (model_error**2 + lc_data[b]["mag_err"]**2)
            + np.log(2. * np.pi * (model_error**2 + lc_data[b]["mag_err"]**2)))

grid[:,0] = lnL

# Get the header for the posterior sample file from the grid file, then save
with open(args.grid_file, "r") as fp:
    header = fp.readline().strip()[2:] # strip off the "# " at the beginning and the "\n" at the end
np.savetxt(args.output_file, grid, header=header)
