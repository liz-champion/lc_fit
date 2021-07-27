#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import argparse
import json

parser = argparse.ArgumentParser(description="Plot light curves")
parser.add_argument("--lc-file", help="JSON file containing the true light curve data")
parser.add_argument("--model-lc-file", help="JSON file containing the model evaluated at the best-fit (or true) parameters")
parser.add_argument("--bands", nargs="+", help="Bands to plot")
parser.add_argument("--output-file", help="Filename to save light curve plot to")
parser.add_argument("--tmin", type=float, default=0.125, help="Minimum time value to plot")
parser.add_argument("--tmax", type=float, default=37., help="Maximum time value to plot")
args = parser.parse_args()

colors = {
    "K":"darkred",
    "H":"red",
    "J":"orange",
    "y":"gold",
    "z":"greenyellow",
    "i":"green",
    "r":"lime",
    "g":"cyan"
}

offsets = { # offset for each band (in mag)
    "K":0,
    "H":1,
    "J":2,
    "y":3,
    "z":4,
    "i":5,
    "r":6,
    "g":7
}

plt.figure(figsize=(12, 8))

if args.lc_file is not None:
    with open(args.lc_file, "r") as fp:
        lc_data = json.load(fp)
    for band in args.bands:
        for key in lc_data[band].keys():
            lc_data[band][key] = np.array(lc_data[band][key])
        plt.errorbar(lc_data[band]["time"], lc_data[band]["mag"] + offsets[band], yerr=lc_data[band]["mag_err"], capsize=4, elinewidth=2, color=colors[band], fmt="none")

if args.model_lc_file is not None:
    with open(args.model_lc_file, "r") as fp:
        lc_data = json.load(fp)
    for band in args.bands:
        for key in lc_data[band].keys():
            lc_data[band][key] = np.array(lc_data[band][key])
        plt.plot(lc_data[band]["time"], lc_data[band]["mag"] + offsets[band], color=colors[band])

plt.xlim(args.tmin, args.tmax)
ticks = [x for x in [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32] if args.tmin <= x <= args.tmax]
labels = map(str, ticks)
plt.gca().set_xscale("log")
plt.gca().set_xticks(ticks)
plt.gca().set_xticklabels(labels, rotation=45)
plt.gca().tick_params(labelsize=16)
plt.gca().invert_yaxis()
plt.ylabel('$m_{AB}$', fontsize=16)
plt.legend(handles=[mlines.Line2D([], [], color=colors[b], marker="o", linestyle="None", markersize=10, label="{0} + {1}".format(b, offsets[b])) for b in args.bands], prop={"size":16})
#plt.legend(prop={"size":16})
plt.xlabel('Time (days)', fontsize=16)
plt.tight_layout()
plt.savefig(args.output_file)
