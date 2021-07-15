#! /usr/bin/env python3

import numpy as np
import argparse
import os
import sys
import json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from scipy.linalg import cholesky, cho_solve

parser = argparse.ArgumentParser(description="Evaluate a single interpolator, specified by `--interp-angle` and `--interp-time`, on the parameters given in `--parameter-file`")
parser.add_argument("--interp-angle", help="Which angle (of 0, 30, 45, 60, 75, 90) to evaluate")
parser.add_argument("--interp-time", help="Which time (of the set values at which the interpolators are trained) to use")
parser.add_argument("--grid-file", help="Filename for the parameter grid")
parser.add_argument("--index-file", help="Name of file containing the indices in the grid that correspond to `--interp-angle`")
parser.add_argument("--output-directory", help="File to write magnitudes to")
parser.add_argument("--band", help="Band to evaluate")
args = parser.parse_args()

# wavelengths corresponding to bands
wavelengths = {
        "g":477.56,
        "r":612.95,
        "i":748.46,
        "z":865.78,
        "y":960.31,
        "J":1235.0,
        "H":1662.0,
        "K":2159.0
}

# _load_gp() and _model_predict() are stolen from Marko's code
def _load_gp(fname_base):
    kernel=None
    with open(fname_base + ".json",'r') as f:
        my_json = json.load(f)
    my_X = np.loadtxt(fname_base + "_X.dat")
    my_y = np.loadtxt(fname_base + "_y.dat")
    my_alpha = np.loadtxt(fname_base + "_alpha.dat")
    dict_params = my_json['kernel_params']
    theta = np.array(my_json['kernel']).astype('float')
    theta = np.power(np.e, theta)
    kernel = WhiteKernel(theta[0]) + theta[1]*RBF(length_scale=theta[2:])
    gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0)
    gp.kernel_ = kernel
    dict_params_eval = {}
    for name in dict_params:
        if not('length' in name   or 'constant' in name):
            continue
        if name =="k2__k2__length_scale":
            one_space = ' '.join(dict_params[name].split())
            dict_params_eval[name] = eval(one_space.replace(' ',','))
        else:
            dict_params_eval[name] = eval(dict_params[name])
    gp.kernel_.set_params(**dict_params_eval)
    gp.X_train_ = my_X
    gp.y_train_ = my_y
    gp.alpha_ = my_alpha
    gp._y_train_std = float(my_json['y_train_std'])
    gp._y_train_mean = float(my_json['y_train_mean'])
    return gp

def _model_predict(model, inputs):
    K = model.kernel_(model.X_train_)
    K[np.diag_indices_from(K)] += model.alpha
    model.L_ = cholesky(K, lower=True) # recalculating L matrix since this is what makes the pickled models bulky
    model._K_inv = None # has to be set to None so the GP knows to re-calculate matrices used for uncertainty
    K_trans = model.kernel_(inputs, model.X_train_)
    pred = K_trans.dot(model.alpha_)
    pred = model._y_train_std * pred + model._y_train_mean
    v = cho_solve((model.L_, True), K_trans.T)
    y_cov = model.kernel_(inputs) - K_trans.dot(v)
    err = np.sqrt(np.diag(y_cov))
    
    mags = _log_lums_to_mags(pred)
    mags_error = 2.5 * err

    return mags, mags_error

def _log_lums_to_mags(log_lums):
    d = 3.086e18 # parsec in cm
    d *= 10 # distance of 10 pc
    log_flux = log_lums - np.log10(4.0 * np.pi * d**2)
    mags = -48.6 - 2.5 * log_flux
    return mags

# load the grid indices, exiting if it has size 0
indices = np.loadtxt(args.index_file).astype(int)
if indices.size == 0:
    exit()

# load the grid
grid = np.loadtxt(args.grid_file)
params = np.empty(grid.shape)
params[:,:4] = grid[:,:4] # take everything but angle
params[:,4] = wavelengths[args.band]
params = params[indices]

# location of trained interpolators
interp_loc = os.environ["INTERP_LOC"]
if interp_loc[-1] != "/":
    interp_loc += "/"
interp_loc += "surrogate_data/2021_Wollaeger_TorusPeanut/"

# load the interpolator
interp_name = interp_loc + "theta" + ("00" if args.interp_angle == "0" else args.interp_angle) + "deg/t_" + args.interp_time + "_days/model"
model = _load_gp(interp_name)

# evaluate the interpolator
mags, mags_err = _model_predict(model, params)

# fill an array with the results
output_array = np.empty((params.shape[0], 2))
output_array[:,0] = mags
output_array[:,1] = mags_err

# write the output to a file
np.savetxt(args.output_directory + ("/" if args.output_directory[-1] != "/" else "") + "eval_interp_{0}_{1}_{2}.dat".format(args.interp_time, args.interp_angle, args.band), output_array)
