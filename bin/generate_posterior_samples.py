#! /usr/bin/env python3

import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Compute log-likelihoods from light curve evaluations")
parser.add_argument("--interp-directory", help="Directory containing the interpolator evaluations and index files")
parser.add_argument("--output-file", help="Name of file to save posterior samples to")
parser.add_argument("--grid-file", help="Location of the grid file")
args = parser.parse_args()

# Load the files containing the grid indices corresponding to each angle
indices_0 = np.loadtxt(args.interp_directory + ("/" if args.directory[-1] != "/" else "") + "indices_0.dat").astype(int)
indices_30 = np.loadtxt(args.interp_directory + ("/" if args.directory[-1] != "/" else "") + "indices_30.dat").astype(int)
indices_45 = np.loadtxt(args.interp_directory + ("/" if args.directory[-1] != "/" else "") + "indices_45.dat").astype(int)
indices_60 = np.loadtxt(args.interp_directory + ("/" if args.directory[-1] != "/" else "") + "indices_60.dat").astype(int)
indices_75 = np.loadtxt(args.interp_directory + ("/" if args.directory[-1] != "/" else "") + "indices_75.dat").astype(int)
indices_90 = np.loadtxt(args.interp_directory + ("/" if args.directory[-1] != "/" else "") + "indices_90.dat").astype(int)

# Load the grid
grid = np.loadtxt(args.grid_file)


