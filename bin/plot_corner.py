#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
import corner

parser = argparse.ArgumentParser(description="Create a corner plot of a posterior distribution")
parser.add_argument("--posterior-file", nargs="+", help="Posterior sample file; can provide multiple arguments to plot several overlaid distributions")
parser.add_argument("--truth-file", help="File containing the true parameter values, if known")
parser.add_argument("--label", nargs="+", help="Labels for distributions; must have the same number of arguments as --posterior-file, assumed to be in the same order")
parser.add_argument("--parameters", nargs="+", help="Parameters to plot")
parser.add_argument("--tempering-exponent", type=float, default=1., help="Tempering exponent to apply to posterior probabilities; ONLY USE FOR TROUBLESHOOTING AS IT CHANGES THE RESULTS")
parser.add_argument("--log-mass", action="store_true", help="Plot log10 of masses")
parser.add_argument("--output-file", help="Filename to save plot to")
parser.add_argument("--grid", action="store_true", help="If plotting a distribution from a grid file, use this to weight samples equally")
args = parser.parse_args()

# A dictionary mapping parameter names to TeX strings, for axis labels
tex_dict = {
        "mej_dyn":"$m_{ej}$ [dyn] $(M_\\odot)$",
        "mej_wind":"$m_{ej}$ [wind] $(M_\\odot)$",
        "log_mej_dyn":"log$_{10}( M_{D} / M_\\odot)$",
        "log_mej_wind":"log$_{10}( M_{W} / M_\\odot)$",
        "vej_dyn":"$v_{D} / c$",
        "vej_wind":"$v_{W} / c$",
        "theta":"$\\theta$ (deg)"
}

# Figure out what parameters we're plotting
parameters_to_plot = []
for parameter in args.parameters:
    if args.log_mass and parameter[:3] == "mej":
        parameters_to_plot.append("log_" + parameter)
    else:
        parameters_to_plot.append(parameter)

# Also generate labels
parameter_labels = [tex_dict[p] if p in tex_dict.keys() else p for p in parameters_to_plot]

# Create a data structure to hold our posterior sample data.
# This is a list of dictionaries; each dictionary represents one posterior sample file, mapping the column names from the file to their respective arrays.
posterior_samples = []
for fname in args.posterior_file:
    sample_dict = {}
    samples = np.loadtxt(fname)
    with open(fname, "r") as fp:
        header = fp.readline().strip().split()[1:] # strip off the '\n' and any spaces, as well as the "#" at the beginning
    for i, column_name in enumerate(header):
        sample_dict[column_name] = samples[:,i]
        if args.log_mass and column_name[:3] == "mej":
            sample_dict["log_" + column_name] = np.log10(sample_dict[column_name])
    sample_dict["label"] = args.label[i] if args.label is not None else None
    posterior_samples.append(sample_dict)

# Take care of truth values, if present. This may seem overly-complicated, but it takes care of --log-mass and also keeps parameters in the right order,
# even if that order differs between the sample files and the truth file
if args.truth_file is not None:
    truths_dict = {}
    truths_array = np.loadtxt(args.truth_file)
    with open(args.truth_file, "r") as fp:
        header = fp.readline().strip().split()[1:] # strip off the '\n' and any spaces, as well as the "#" at the beginning
    for i, column_name in enumerate(header):
        truths_dict[column_name] = truths_array[i]
        if args.log_mass and column_name[:3] == "mej":
            truths_dict["log_" + column_name] = np.log10(truths_dict[column_name])
    truths = [truths_dict[p] for p in parameters_to_plot]
else:
    truths = None

# Now, finally, make the plot
fig = None
for i, sample_dict in enumerate(posterior_samples):
    # Fill a 2d array with samples
    to_plot = np.empty((sample_dict["ln(L)"].size, len(parameters_to_plot)))
    for j, parameter in enumerate(parameters_to_plot):
        to_plot[:,j] = sample_dict[parameter]
    # Compute the sample weights
    if args.grid:
        weights = np.ones(sample_dict["ln(L)"].size) / sample_dict["ln(L)"].size
    else:
        ln_L = sample_dict["ln(L)"]
        ln_L = (ln_L - np.max(ln_L)) * args.tempering_exponent
        log_weights = ln_L + sample_dict["ln(p)"] - sample_dict["ln(ps)"]
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)
    # Strip out samples with weight 0
    mask = weights > 0.
    to_plot = to_plot[mask]
    weights = weights[mask]
    print("{0} samples with weight > 0".format(weights.size))
    # Plot the samples
    fig = corner.corner(to_plot, weights=weights, fig=fig, truths=truths, plot_datapoints=False, labels=parameter_labels, label_kwargs={"fontsize":16})

plt.savefig(args.output_file)
