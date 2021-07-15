#! /usr/bin/env python3

import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Takes a grid of parameter samples and splits it based on which angular bin a given sample falls inside of")
parser.add_argument("--grid-file", help="Name of the full grid file")
parser.add_argument("--output-directory", help="Directory to save partitioned grid indices to")
args = parser.parse_args()

grid = np.loadtxt(args.grid_file)
theta = grid[:,4]

indices_0 = np.where(theta < 30.)[0]
indices_30 = np.where(theta < 60.)[0]
indices_45 = np.where((30. <= theta) & (theta < 60.))[0]
indices_60 = np.where((45. <= theta) & (theta < 75.))[0]
indices_75 = np.where(60. <= theta)[0]
indices_90 = np.where(75. <= theta)[0]

fname = args.output_directory + ("/" if args.output_directory[-1] != "/" else "") + "indices"

np.savetxt("{0}_{1}.dat".format(fname, 0), indices_0)
np.savetxt("{0}_{1}.dat".format(fname, 30), indices_30)
np.savetxt("{0}_{1}.dat".format(fname, 45), indices_45)
np.savetxt("{0}_{1}.dat".format(fname, 60), indices_60)
np.savetxt("{0}_{1}.dat".format(fname, 75), indices_75)
np.savetxt("{0}_{1}.dat".format(fname, 90), indices_90)
