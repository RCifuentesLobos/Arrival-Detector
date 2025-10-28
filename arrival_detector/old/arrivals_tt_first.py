import HySEAio as hio
import detectutils as du
import POIs as poi
import greatcircle as grc
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
import copy

"""
Core function of the arrival detection algorithm
Kemal Firdaus
Rodrigo Cifuentes Lobos
"""

# ----------------------------------------------------------------
# 1) List output files and read them
# ----------------------------------------------------------------
# directory with output files
dir_out: str = "/Users/rodrigocifuentes/Documents/Alemania/Universidad/Publications/Tsunami Hazard/Data/test"
# list all output files
files = hio.list_outputs(dir_out)
# read data
time, lon, lat, \
    eta, mask_time, mask_lon, mask_lat, mask_eta, \
    fault_info = hio.read_outputs(files[0])

# get only valid data
inv_idx, valid_eta = hio.detect_invalid_values(mask_eta, eta)
# get valid coordinates
vlon, vlat = hio.get_valid_coordinates(lon, lat, inv_idx)
# get fault information
fault_info_list = hio.get_fault_attributes(fault_info)
# fault lon and lat
flon: float = fault_info_list[0]['lon_barycenter']
flat: float = fault_info_list[0]['lat_barycenter']
# Get length of time series and number of POIs
timelen: int = len(time)
npois: int = np.shape(valid_eta)[1]


# ----------------------------------------------------------------
# 4) Compute the distance between fault and POI
# ----------------------------------------------------------------
# initialize array with distances and travel times
dist_tt = np.zeros((npois, 2))
# loop through all POIs
for i, (lo, la) in enumerate(zip(vlon, vlat)):
    dist_tt[i, 0] = grc.haversine(lo, la, flon, flat)
    dist_tt[i, 1] = grc.estimate_travel_time(lo, la, flon, flat)
# Get index
# time interval in seconds between elevation computation
deltat = 30
arrival_idx = np.floor(dist_tt[:, 1]/deltat).astype(int)


# ----------------------------------------------------------------
# 3) Check if detection is correct
# ----------------------------------------------------------------
# initialize the boolean array
nborders = 3
arrival_status = np.zeros((len(arrival_idx), nborders), dtype=bool)

for i, idx in enumerate(arrival_idx):
    arrival_status[i, :] = du.check_arrival_detection(valid_eta[:, i],
                                                      idx, n=nborders)

# ----------------------------------------------------------------
# 2) Compute STA/LTA for POIs if detection is incorrect
# ----------------------------------------------------------------
# initialize array with the detection values
detection_idx = np.zeros(npois)
# define sta/lta and detection parameters
sta_window = 20
lta_window = 600
trigger_value = 8
# loop through all POIs
for i in range(npois):
    # check if detecion wasn't correct
    if arrival_status[i, :].all():
        # compute sta/lta
        _, _, stalta = du.sta_lta(valid_eta[:, i], sta_window, lta_window)
        # get detection values
        _, detection_idx[i] = du.trigger(stalta, threshold=trigger_value)
# transform to integer
detection_idx = [i.astype(int) for i in detection_idx]


# ----------------------------------------------------------------
# 5) Check which value is better (the bigger, the more zeros are cropped)
# ----------------------------------------------------------------
final_idx = du.get_better_estimate(arrival_idx, detection_idx,
                                   arrival_status)


# ----------------------------------------------------------------
# 6) Check if the detection was successful
# ----------------------------------------------------------------
# initialize the boolean array for the detection status after
# checking the best estimate
detection_status_after = np.zeros((len(final_idx), nborders), dtype=bool)
# Loop through all detection status
for i, idx in enumerate(final_idx):
    detection_status_after[i, :] = du.check_arrival_detection(valid_eta[:, i],
                                                              idx, n=nborders)


# ----------------------------------------------------------------
# 7) Correct the detection status if it wasn't successful
# ----------------------------------------------------------------
for i, status in enumerate(detection_status_after):
    # if the detection was unsuccessful (i.e. the detection status is not all Trues)
    if not status.all():
        # we correct the detection
        final_idx[i] = du.correct_detection(
            valid_eta[:, i], final_idx[i], movement=2)
