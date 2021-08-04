# Bayesian parameter inference on kilonova light curves with `lc_fit`

An in-depth description of the methods can be found [here](https://github.com/liz-champion/lc_fit/blob/main/doc/tex/lc_fit.pdf)

## Installation

It is currently not necessary to install `lc_fit` with pip, as it consists of separate scripts and is not structured as a Python package. However, it is necessary to set two environment variables. First, clone the repository:

```
$ git clone https://github.com/liz-champion/lc_fit.git
```

You'll also need the interpolated light curve model:

```
$ git clone https://github.com/markoris/surrogate_kne.git
```

Next, set the `LC_FIT_LOCATION` environment variable to the location of the `lc_fit` repository; for example, if you cloned the respository in your home directory, add the following to your `.bashrc`, substituting your own username:

```
export LC_FIT_LOCATION=/home/user.name/lc_fit
```

Similarly, set `INTERP_LOC` to point to the location of the interpolator repository. Make sure to `source` your `.bashrc` before continuing:

```
$ source ~/.bashrc
```

## Usage

The Makefile provided with the `lc_fit` repository can set up several types of parameter estimation runs. At time of writing, these are:
- `test`
- `simulation`
- `GW170817`

To demonstrate the usage of the code, we'll walk through a simulation injection/recovery test. To set up the test, run

```
$ make simulation
```

### What did this do?

A simulation injection/recovery test takes simulation data similar to that used to train the GP interpolators, but for parameters that were not used in training. The Makefile first sets up the proper directory structure:
- `pe_runs/` (directory where all PE runs are set up)
    - `simulation_[DATE]/` (each PE run is set up in a directory labeled by date; you can specify an additional label with the `make` command, e.g. `$ make simulation LABEL=_a`)
        - `interp_files/` (directory in which all interpolated light curve data is stored)
        - `condor_logs/` (directory for HTCondor log files)

Next, an initial parameter grid is sampled according to each parameter's prior distribution using `generate_initial_grid.py`. Following this, the simulation file is read, interpolated in time, and saved as a JSON file as the input light curve data for the PE run. Then `generate_initial_grid.py` is used once more, this time to save a grid containing a single point which represents the injection parameters. Finally, `dag_setup.py` is used to create all the necessary HTCondor `.sub` files and the `.dag` file itself.

Note that some injection parameters can be modified in the Makefile, namely the angular bin from which to pull simulation data, the minimum and maximum time values, the number of data points per wavelength band, and the standard deviation of the Gaussian noise added to the data.

Look in the run directory created by the Makefile. It contains a number of `.sub` files, the `.dag` file, the `lc_data.json` injection data file, the injection grid, and the initial sample grid. To launch the PE run, do

```
$ condor_submit_dag run.dag
```

The run's progress can be monitored with `condor_q` or `condor_watch_q`. As the run progresses, posterior sample files named `posterior-samples_N.dat` should appear for each iteration `N`, as well as new grids named `grid_N.dat`.

To understand how the algorithm works and what information all the files that are produced contain, read the documentation linked above. Also note that the contents of the `condor_logs/` directory are particularly useful for troubleshooting.

To plot our results, use the plotting script located in `lc_fit/bin/`. For our test, the following command will generate a corner plot containing all the parameters, with masses on a log scale and vertical lines indicating the true parameter values:

```
$ python3 ${LC_FIT_LOCATION}/bin/plot_corner.py --log-mass --truth-file injection_grid.dat --parameters mej_dyn mej_wind vej_dyn vej_wind theta --posterior-file posterior_samples_N.dat --output-file corner.png
```

Two command line arguments for this script are helpful when troubleshooting a PE run:
- The `--grid` argument will cause the sample weights and priors to be ignored, instead plotting the underlying sampling distribution.
- The `--tempering-exponent` argument can be used to supply a power to which the likelihood should be raised (see Section 4 of the writeup linked above). Note however that **this argument is for debugging purposes only**: it artifically flattens and broadens the posterior distribution, hence the plot it generates is not an accurate representation of the results.

## Custom PE runs

Documentation for each Python script in `lc_fit` is coming soon; meanwhile, the Makefile provides a useful starting point for setting up custom runs.