#! /usr/bin/env python3

import numpy as np
from scipy.stats import loguniform, truncnorm
import argparse

parser = argparse.ArgumentParser(description="Generate initial (random) sample grid")
parser.add_argument("--fixed-parameter", nargs=2, action="append", help="Fix a parameter's value, e.g. `--fixed-parameter dist 40.0`")
parser.add_argument("--set-limit", nargs=3, action="append", help="Set a parameter's limits to something other than the default, e.g. `--set-limit mej_dyn 0.01, 0.05`")
parser.add_argument("--npts", type=int, default=25000, help="Number of points to use in the grid")
parser.add_argument("--output-file", help="Filename to save grid to")
parser.add_argument("--gaussian-prior", action="append", nargs=3, help="Give a parameter a Gaussian prior, specifying the mean and standard deviation; for example, `--gaussian-prior theta 20.0 5.0`")
args = parser.parse_args()

#
# Prior functions
#

def uniform(llim, rlim, x):
    return 1. / (rlim - llim)

def log_uniform(llim, rlim, x):
    return loguniform.pdf(x, llim, rlim)

def gaussian(llim, rlim, mu, sigma, x):
    return truncnorm.pdf(x, llim, rlim, loc=mu, scale=sigma)
   
#
# Functions to draw samples from the priors
#
def sample_uniform(llim, rlim, n):
    return np.random.uniform(llim, rlim, size=n)

def sample_log_uniform(llim, rlim, n):
    return loguniform.rvs(llim, rlim, size=n)

def sample_gaussian(llim, rlim, mu, sigma, n):
    return truncnorm.rvs(llim, rlim, loc=mu, scale=sigma, size=n)

# the parameters FIXME: should this be more flexible?
ordered_parameters = ["mej_dyn", "vej_dyn", "mej_wind", "vej_wind", "theta"]

# Parameter limits
limits = {
        "mej_dyn":[0.001, 0.1],
        "vej_dyn":[0.05, 0.3],
        "mej_wind":[0.001, 0.1],
        "vej_wind":[0.05, 0.3],
        "theta":[0., 90.]
}

# If the user specified different limits, change them accordingly
if args.set_limit is not None:
    for [_parameter, _llim, _rlim] in args.set_limit:
        limits[_parameter] = [float(_llim), float(_rlim)]


# Specify each parameter's prior
prior_functions = {
        "mej_dyn":lambda x: log_uniform(*limits["mej_dyn"], x),
        "vej_dyn":lambda x: uniform(*limits["vej_dyn"], x),
        "mej_wind":lambda x: log_uniform(*limits["mej_wind"], x),
        "vej_wind":lambda x: uniform(*limits["vej_wind"], x),
        "theta":lambda x: uniform(*limits["theta"], x)
}

# Specify each parameter's prior sampling function
prior_sampling_functions = {
        "mej_dyn":lambda n: sample_log_uniform(*limits["mej_dyn"], n),
        "vej_dyn":lambda n: sample_uniform(*limits["vej_dyn"], n),
        "mej_wind":lambda n: sample_log_uniform(*limits["mej_wind"], n),
        "vej_wind":lambda n: sample_uniform(*limits["vej_wind"], n),
        "theta":lambda n: sample_uniform(*limits["theta"], n)
}

# Deal with possible fixed parameters
fixed_parameters = {}
if args.fixed_parameter is not None:
    for [_parameter, _val] in args.fixed_parameter:
        fixed_parameters[_parameter] = float(_val)

# Gaussian priors
if args.gaussian_prior is not None:
    for [_parameter, _mu, _sigma] in args.gaussian_prior:
        _mu, _sigma = float(_mu), float(_sigma)
        a, b = (limits[_parameter][0] - _mu) / _sigma, (limits[_parameter][1] - _mu) / _sigma
        prior_functions[_parameter] = lambda x: gaussian(a, b, _mu, _sigma, x)
        prior_sampling_functions[_parameter] = lambda n: sample_gaussian(a, b, _mu, _sigma, n)

#
# Generate the grid
#

grid = np.empty((args.npts, len(ordered_parameters) + 3))

# The first column, for lnL, gets filled in later (by generate_posterior_samples.py), so for now make it 0
grid[:,0] = 0.

# The second and third columns are the prior and sampling prior, respectively, which are the same for the initial grid.
# The joint prior is the product of all the separate priors, so we'll set them to 1 now and multiply them by each parameter's prior in the loop.
grid[:,1] = 0.

# Do the sampling and compute priors
for i, _parameter in enumerate(ordered_parameters):
    grid[:,i + 3] = prior_sampling_functions[_parameter](args.npts) if _parameter not in fixed_parameters.keys() else fixed_parameters[_parameter]
    grid[:,1] += np.log(prior_functions[_parameter](grid[:,i + 3]))
grid[:,2] = grid[:,1]

# Save the grid
np.savetxt(args.output_file, grid, header=("ln(L) ln(p) ln(ps) " + " ".join(ordered_parameters)))
