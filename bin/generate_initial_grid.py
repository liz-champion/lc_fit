#! /usr/bin/env python3

import numpy as np
from scipy.stats import loguniform, truncnorm
import argparse

parser = argparse.ArgumentParser(description="Generate initial (random) sample grid")
parser.add_argument("--fixed-parameter", nargs=2, action="append", help="Fix a parameter's value, e.g. `--fixed-parameter dist 40.0`")
parser.add_argument("--set-limit", nargs=3, action="append", help="Set a parameter's limits to something other than the default, e.g. `--set-limit mej_dyn 0.01, 0.05`")
parser.add_argument("--npts", type=int, default=10000, help="Number of points to use in the grid")
parser.add_argument("--output-file", help="Filename to save grid to")
# TODO: implement --gaussian-prior option
args = parser.parse_args()

#
# functions to draw samples from the possible priors
#
def sample_uniform(llim, rlim, n):
    return np.random.uniform(llim, rlim, size=n)

def sample_log_uniform(llim, rlim, n):
    return loguniform.rvs(llim, rlim, size=n)

def sample_gaussian(llim, rlim, mu, sigma, n):
    return truncnorm.rvs(llim, rlim, loc=mu, scale=sigma, size=n)
   
# the parameters FIXME: should this be more flexible?
ordered_parameters = ["mej_dyn", "vej_dyn", "mej_wind", "vej_wind", "theta"]

# parameter limits
limits = {
        "mej_dyn":[0.001, 0.1],
        "vej_dyn":[0.05, 0.3],
        "mej_wind":[0.001, 0.1],
        "vej_wind":[0.05, 0.3],
        "theta":[0., 90.]
}

# if the user specified different limits, change them accordingly
if args.set_limit is not None:
    for [_parameter, _llim, _rlim] in args.set_limit:
        limits[_parameter] = [float(_llim), float(_rlim)]

# specify each parameter's prior
priors = {
        "mej_dyn":lambda n: sample_log_uniform(*limits["mej_dyn"], n),
        "vej_dyn":lambda n: sample_uniform(*limits["vej_dyn"], n),
        "mej_wind":lambda n: sample_log_uniform(*limits["mej_wind"], n),
        "vej_wind":lambda n: sample_uniform(*limits["vej_wind"], n),
        "theta":lambda n: sample_uniform(*limits["theta"], n)
}

# deal with possible fixed parameters
fixed_parameters = {}
if args.fixed_parameter is not None:
    for [_parameter, _val] in args.fixed_parameter:
        fixed_parameters[_parameter] = float(_val)

# do the sampling
grid = np.empty((args.npts, len(ordered_parameters)))
for i, _parameter in enumerate(ordered_parameters):
    grid[:,i] = priors[_parameter](args.npts) if _parameter not in fixed_parameters.keys() else fixed_parameters[_parameter]

# save the grid
np.savetxt(args.output_file, grid, header=" ".join(ordered_parameters))
