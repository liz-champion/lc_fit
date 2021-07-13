#! /usr/bin/env python3

import numpy as np
import argparse
import json
from astropy.time import Time

predefined_bands = ["g", "r", "i", "z", "y", "J", "H", "K"]


#
# command line arguments
#
parser = argparse.ArgumentParser(description="Parse Open Astronomy Catalog (OAC) JSON files")
parser.add_argument("--t0", type=float, default=0, help="Initial time (t=0 for event)")
parser.add_argument("--json-file", help="Filename for JSON file")
parser.add_argument("--bands", nargs="+", help="Data bands to store")
parser.add_argument("--output-file", help="Filename to write output JSON to")
parser.add_argument("--tmax", type=float, default=np.inf, help="Upper bound for time points to keep")
parser.add_argument("--time-format", type=str, default="gps", help="Time format (MJD or GPS)")
parser.add_argument("--telescopes", action='append', nargs="+", help="Telescopes to use (defaults to all)")
for b in predefined_bands:
    parser.add_argument("--tmax-" + b, type=float, help="Upper bound for time in " + b + " band")
args = parser.parse_args()

# generally t0 will be in GPS time, but we want it in MJD
t0 = args.t0
if args.time_format == "gps":
    t0 = Time(t0, format="gps").mjd

# create a set containing the names of telescopes to use
if args.telescopes is not None:
    telescopes = set(args.telescopes)

# determine event name from JSON file name
name = args.json_file.split('/')[-1] # get rid of path except for filename
name = name.split('.')[0] # get event name from filename

# read in the data
with open(args.json_file, "r") as fp:
    data = json.load(fp, encoding="UTF-8")[name]["photometry"]

# create empty data arrays
data_dict = {b:{"time":[], "mag":[], "mag_err":[]} for b in args.bands}

# iterate over the entries in the JSON file
for entry in data:
    if "band" in entry:
        band = entry["band"]
        # check a whole bunch of conditions to see if we care about this data point
        if (band in args.bands and "e_magnitude" in entry and "telescope" in entry and "source" in entry
            and (args.telescopes is None or entry["telescope"] in telescopes)
            and "realization" not in entry):
            # check if the data point is before tmax AND tmax_[band]
            tmax_here = args.tmax
            if "tmax_" + band in vars(args).keys() and vars(args)["tmax_" + band] is not None:
                tmax_here = min(tmax, vars(args)["tmax_" + band])
            # add the data point to the output dictionary
            if float(entry["time"]) - t0 < tmax_here:
                data_dict[band]["time"].append(float(entry["time"]) - t0)
                data_dict[band]["mag"].append(float(entry["magnitude"]))
                data_dict[band]["mag_err"].append(float(entry["e_magnitude"]))

with open(args.output_file, "w") as fp:
    json.dump(data_dict, fp, indent=4)
