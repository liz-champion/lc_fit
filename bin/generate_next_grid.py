#! /usr/bin/env python3

import numpy as np
from scipy.stats import loguniform, truncnorm
import argparse
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

parser = argparse.ArgumentParser(description="Fit a Gaussian Mixture Model to posterior samples and sample from the model to generate the next grid")
parser.add_argument("--posterior-file", help="Location of the posterior sample file")
parser.add_argument("--output-file", help="Filename to save grid to")
parser.add_argument("--npts", type=int, default=10000, help="Number of points to use in the grid")
parser.add_argument("--set-limit", nargs=3, action="append", help="Set a parameter's limits to something other than the default, e.g. `--set-limit mej_dyn 0.01, 0.05`")
parser.add_argument("--fixed-parameter", nargs=2, action="append", help="Fix a parameter's value, e.g. `--fixed-parameter dist 40.0`. NOT IMPLEMENTED")
parser.add_argument("--tempering-exponent", type=float, default=1., help="Exponent (between 0 and 1) to be applied to the likelihoods for the fit. Helps with the first few iterations where very few points have nonzero likelihoods")
parser.add_argument("--n-procs-kde", type=int, default=4, help="Number of parallel processes to use for fitting KDEs when testing for the best bandwidth")
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
        "mej_dyn":lambda n: log_uniform(*limits["mej_dyn"], n),
        "vej_dyn":lambda n: uniform(*limits["vej_dyn"], n),
        "mej_wind":lambda n: log_uniform(*limits["mej_wind"], n),
        "vej_wind":lambda n: uniform(*limits["vej_wind"], n),
        "theta":lambda n: uniform(*limits["theta"], n)
}

# Deal with possible fixed parameters TODO take care of this, make sure fixed parameters are not included in the KDE
#fixed_parameters = {}
#if args.fixed_parameter is not None:
#    for [_parameter, _val] in args.fixed_parameter:
#        fixed_parameters[_parameter] = float(_val)

# The parameters we need to fit and sample
#parameters_used = [_parameter for _parameter in ordered_parameters if parameter not in fixed_parameters.keys()]

# Convert masses to log10(mass), then scale parameters to be in the interval [0, 1]
def transform(parameters):
    transformed_parameters = np.empty(parameters.shape)
    for i, _parameter in enumerate(ordered_parameters):
        llim, rlim = limits[_parameter]
        if _parameter[:3] == "mej":
            llim, rlim = np.log10(llim), np.log10(rlim)
            transformed_parameters[:,i] = (np.log10(parameters[:,i]) - llim) / (rlim - llim)
        else:
            transformed_parameters[:,i] = (parameters[:,i] - llim) / (rlim - llim)
    return transformed_parameters 

def inverse_transform(parameters):
    transformed_parameters = np.empty(parameters.shape)
    for i, _parameter in enumerate(ordered_parameters):
        llim, rlim = limits[_parameter]
        if _parameter[:3] == "mej":
            llim, rlim = np.log10(llim), np.log10(rlim)
            transformed_parameters[:,i] = 10.**(parameters[:,i] * (rlim - llim) + llim)
        else:
            transformed_parameters[:,i] = parameters[:,i] * (rlim - llim) + llim
    return transformed_parameters 

# Load the posterior samples
samples = np.loadtxt(args.posterior_file)

# Compute sample weights
ln_L = samples[:,0]
ln_p = samples[:,1]
ln_ps = samples[:,2]
ln_L = (ln_L - np.max(ln_L)) * args.tempering_exponent
log_weights = ln_L + ln_p - ln_ps
log_weights -= np.max(log_weights)
weights = np.exp(log_weights)
weights /= np.sum(weights) # normalize the weights

# Do the scaling of the parameters
parameters = transform(samples[:,3:])

# Strip out samples with weight NaN
parameters = parameters[np.isfinite(weights) & (weights > 0)]
weights = weights[np.isfinite(weights) & (weights > 0)]

# How many do we have left? Useful for diagnosing what's gone wrong
print("{0} samples with finite, positive weights".format(weights.size))

# We want to fit a KDE to the posterior samples so we can sample from it.
# The fit is quite sensitive to a hyperparameter called bandwidth, which specifies the width of the (in this case, Gaussian) kernel.
# Fortunately, scikit-learn has a built-in way to optimize this sort of hyperparameter, GridSearch.
hyperparameter_grid = GridSearchCV(KernelDensity(kernel="gaussian"), {"bandwidth":np.logspace(-3., -0.5, 10)}, cv=5, n_jobs=args.n_procs_kde)
hyperparameter_grid.fit(parameters, sample_weight=weights)
bandwidth = hyperparameter_grid.best_estimator_.bandwidth
print("Using bandwidth = {0}".format(bandwidth))

# Since GridSearch splits the data into train and test sets, we don't want to use the trained KDEs from this process.
# Instead, take the optimal bandwidth and retrain with it on the full data set
kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
kde.fit(parameters, sample_weight=weights)

# Now we want to sample from the KDE, being sure to thow out points that lie outside the bounds
new_samples = np.empty((0, len(ordered_parameters)))
llim_array = np.array([limits[_parameter][0] for _parameter in ordered_parameters])
rlim_array = np.array([limits[_parameter][1] for _parameter in ordered_parameters])
while new_samples.shape[0] < args.npts:
    new_samples_here = inverse_transform(kde.sample(args.npts))
    new_samples_here = new_samples_here[np.where(np.all(new_samples_here > llim_array, axis=1) & np.all(new_samples_here < rlim_array, axis=1))]
    new_samples = np.append(new_samples, new_samples_here, axis=0)

# Keep only the number of samples requested by the user
new_samples = new_samples[:args.npts]

#
# Generate the new grid
#

grid = np.empty((args.npts, len(ordered_parameters) + 3))
grid[:,3:] = new_samples

# The first column, for ln_L, gets filled in later (by generate_posterior_samples.py), so for now make it 0
grid[:,0] = 0.

# The second and third columns are the prior and sampling prior, respectively.
# The joint prior is the product of all the separate priors, so we'll set them to 1 now and multiply them by each parameter's prior in the loop.
grid[:,1] = 0.

# Do the sampling and compute (log) priors
for i, _parameter in enumerate(ordered_parameters):
    grid[:,1] += np.log(prior_functions[_parameter](grid[:,i + 3]))

# Compute the sampling prior from the KDE (note this returns log probability)
grid[:,2] = kde.score_samples(new_samples)

# Save the grid
np.savetxt(args.output_file, grid, header=("ln(L) ln(p) ln(ps) " + " ".join(ordered_parameters)))
