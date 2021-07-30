#! /usr/bin/env python3

import numpy as np
from scipy.stats import loguniform, truncnorm, multivariate_normal
import argparse
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

parser = argparse.ArgumentParser(description="Fit a Gaussian Mixture Model to posterior samples and sample from the model to generate the next grid")
parser.add_argument("--posterior-file", help="Location of the posterior sample file")
parser.add_argument("--output-file", help="Filename to save grid to")
parser.add_argument("--npts", type=int, default=10000, help="Number of points to use in the grid")
parser.add_argument("--set-limit", nargs=3, action="append", help="Set a parameter's limits to something other than the default, e.g. `--set-limit mej_dyn 0.01, 0.05`")
parser.add_argument("--tempering-exponent", type=float, default=1., help="Exponent (between 0 and 1) to be applied to the likelihoods for the fit. Helps with the first few iterations where very few points have nonzero likelihoods")
parser.add_argument("--n-procs-kde", type=int, default=4, help="Number of parallel processes to use for fitting KDEs when testing for the best bandwidth")
parser.add_argument("--gaussian-sampler", action="store_true", help="Use a multivariate Gaussian fit to the data to generate the next grid rather than a KDE")
parser.add_argument("--fixed-parameters", nargs="+", help="Parameters that stay fixed to their grid values")
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

fixed_parameters = args.fixed_parameters if args.fixed_parameters is not None else []

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

# Convert masses to log10(mass), then scale parameters to be in the interval [0, 1]
# This function also removes fixed parameters from the returned array
def transform(parameters):
    transformed_parameters = np.empty((parameters.shape[0], len(ordered_parameters) - len(fixed_parameters)))
    i = 0
    for _parameter in ordered_parameters:
        if _parameter in fixed_parameters:
            continue
        llim, rlim = limits[_parameter]
        if _parameter[:3] == "mej":
            llim, rlim = np.log10(llim), np.log10(rlim)
            transformed_parameters[:,i] = (np.log10(parameters[:,i]) - llim) / (rlim - llim)
        else:
            transformed_parameters[:,i] = (parameters[:,i] - llim) / (rlim - llim)
        i += 1
    return transformed_parameters 

# Inverts the above transformation
def inverse_transform(parameters):
    transformed_parameters = np.empty((parameters.shape[0], len(ordered_parameters) - len(fixed_parameters)))
    i = 0
    for _parameter in ordered_parameters:
        if _parameter in fixed_parameters:
            continue
        llim, rlim = limits[_parameter]
        if _parameter[:3] == "mej":
            llim, rlim = np.log10(llim), np.log10(rlim)
            transformed_parameters[:,i] = 10.**(parameters[:,i] * (rlim - llim) + llim)
        else:
            transformed_parameters[:,i] = parameters[:,i] * (rlim - llim) + llim
        i += 1
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

original_parameters = samples[:,3:]
parameters = transform(samples[:,3:])

# Strip out samples with weight NaN
parameters = parameters[np.isfinite(weights) & (weights > 0)]
weights = weights[np.isfinite(weights) & (weights > 0)]

# How many do we have left? Useful for diagnosing what's gone wrong
print("{0} samples with finite, positive weights".format(weights.size))

sampler = None
if args.gaussian_sampler:
    # Compute the mean and covariance of our data, adding a bit to the diagonal of the covariance matrix to 
    mu = np.average(parameters, weights=weights, axis=0)
    cov = np.cov(parameters, rowvar=False, aweights=weights)# + np.eye(mu.size) * 0.1 # FIXME probably there should be some systematic way of choosing this instead of hard-coding to 0.1

    if np.linalg.matrix_rank(cov) < cov.shape[0]:
        # If this happens, we got a singular matrix, which will cause the sampling to fail.
        print("Got singular matrix, stripping out off-diagonal elements")
        diag = np.diag(cov)
        cov = np.zeros(cov.shape)
        np.fill_diagonal(cov, diag)

    # Make a function for sampling new points
    sampler = lambda n: multivariate_normal.rvs(mean=mu, cov=cov, size=n)

    # Make a function for the sampling prior
    sampling_prior = lambda x: multivariate_normal.logpdf(x, mean=mu, cov=cov)

else:
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

    # Make a function for sampling new points
    sampler = kde.sample

    # Make a function for the sampling prior
    sampling_prior = lambda x: kde.score_samples(x)

# Now we want to generate new samples, being sure to thow out points that lie outside the bounds
new_samples = np.empty((0, len(ordered_parameters) - len(fixed_parameters)))
llim_array = np.array([limits[_parameter][0] for _parameter in ordered_parameters if _parameter not in fixed_parameters])
rlim_array = np.array([limits[_parameter][1] for _parameter in ordered_parameters if _parameter not in fixed_parameters])
while new_samples.shape[0] < args.npts:
    new_samples_here = inverse_transform(sampler(args.npts))
    new_samples_here = new_samples_here[np.where(np.all(new_samples_here > llim_array, axis=1) & np.all(new_samples_here < rlim_array, axis=1))]
    new_samples = np.append(new_samples, new_samples_here, axis=0)

# Keep only the number of samples requested by the user
new_samples = new_samples[:args.npts]

#
# Generate the new grid
#

grid = np.empty((args.npts, len(ordered_parameters) + 3))

# Put the newly sampled points in the grid, carrying fixed parameters over from their original values
j = 0
for i, _parameter in enumerate(ordered_parameters):
    if _parameter in fixed_parameters:
        grid[:,i + 3] = original_parameters[:,i][0] # Take the first value from the corresponding column in the original samples rather than copying the whole column,
                                                    # since in principle the new grid could have a different number of rows than the input.
    else:
        grid[:,i + 3] = new_samples[:,j]
        j += 1

# The first column, for ln_L, gets filled in later (by generate_posterior_samples.py), so for now make it 0
grid[:,0] = 0.

# The second and third columns are the prior and sampling prior, respectively.
# The joint prior is the product of all the separate priors, so we'll set them to 1 now and multiply them by each parameter's prior in the loop.
grid[:,1] = 0.

# Do the sampling and compute (log) priors
for i, _parameter in enumerate(ordered_parameters):
    grid[:,1] += np.log(prior_functions[_parameter](grid[:,i + 3]))

# Compute the sampling prior from the Gaussian or KDE (note that both return log probability)
# NOTE regarding normalization: technically, these sampling priors are not normalized since neither the Gaussian fit nor the KDE cares about the bounds of our parameter space.
# However, it doesn't actually matter - these probabilities are used to calculate sample weights that are then normalized, meaning any constant factor is inconsequential.
grid[:,2] = sampling_prior(transform(new_samples)) # Side note for future readers: this line cost me several days of effort searching for a bug.
                                                   # I forgot to transform the samples before calculating the sampling prior (since the Gaussian/KDE was fit to transformed data),
                                                   # meaning I was getting horrible garbage for a reason that took me way too long to figure out. *sigh*...

# Save the grid
np.savetxt(args.output_file, grid, header=("ln(L) ln(p) ln(ps) " + " ".join(ordered_parameters)))
