#! /usr/bin/env python3

import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Fit a Gaussian Mixture Model to posterior samples and sample from the model to generate the next grid")
parser.add_argument("--posterior-file", help="Location of the posterior sample file")
parser.add_argument("--output-file", help="Filename to save grid to")
args = parser.parse_args()

# Load the posterior samples
samples = np.loadtxt(args.posterior_file)


