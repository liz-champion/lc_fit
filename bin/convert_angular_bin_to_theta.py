import numpy as np
import argparse

parser = argparse.ArgumentParser(description="A silly script to convert the angular bin of a light curve in the simulation data to its angle in degrees")
parser.add_argument("--angular-bin", type=int, help="The angular bin")
args = parser.parse_args()

# Averge the upper and lower bounds of the angular bin
theta = np.degrees(0.5 * (np.arccos(1. - (2. * args.angular_bin / 54.)) + np.arccos(1. - (2. * (args.angular_bin - 1) / 54.))))

print(theta)
