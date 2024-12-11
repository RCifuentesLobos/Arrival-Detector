import HySEAio as hio
import detectutils as du
import POIs as poi
import greatcircle as grc
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
import copy


# directory with output files
dir_out: str = "/Users/rodrigocifuentes/Documents/Alemania/Universidad/Publications/Tsunami Hazard/Data/test"
# time interval in seconds between elevation computation
deltat = 30
# define sta/lta and detection parameters
sta_window = 20
lta_window = 600
trigger_value = 8


def main(dir_out: str = dir_out,
         deltat: float = deltat,
         sta_window: int = sta_window,
         lta_window: int = lta_window,
         trigger_value: int = trigger_value) -> None:
    """
    Parameters:
    dir_out: str. The directory with .nc outputs
    deltat: float. The time interval in seconds between succesive
      elevation data points
    sta_window: int. The length of the STA window
    lta_window: int. The length of the LTA window
    trigger_value: int. Threshold for detecting an event
    """
    # list all output files
    files = hio.list_outputs(dir_out)
    # loop through all output files
    for file in files:
        # ----------------------------------------------------------------
        # 1) List output files and read them
        # ----------------------------------------------------------------
        # read data
        time, lon, lat, \
            eta, mask_time, mask_lon, mask_lat, mask_eta, \
            fault_info = hio.read_outputs(file)
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
        # 2) Compute the distance between fault and POI
        # ----------------------------------------------------------------
        # initialize array with distances and travel times
        dist_tt = np.zeros((npois, 2))
        # loop through all POIs
        for i, (lo, la) in enumerate(zip(vlon, vlat)):
            dist_tt[i, 0] = grc.haversine(lo, la, flon, flat)
            dist_tt[i, 1] = grc.estimate_travel_time(lo, la, flon, flat)
        # Get index
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
        # 4) Compute STA/LTA for POIs if detection is incorrect
        # ----------------------------------------------------------------
        # initialize array with the detection values
        detection_idx = np.zeros(npois)
        # loop through all POIs
        for i in range(npois):
            # check if detecion wasn't correct
            if not arrival_status[i, :].all():
                # compute sta/lta
                _, _, stalta = du.sta_lta(
                    valid_eta[:, i], sta_window, lta_window)
                # get detection values
                _, detection_idx[i] = du.trigger(
                    stalta, threshold=trigger_value)
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
        detection_status_after = np.zeros((len(final_idx),
                                           nborders), dtype=bool)
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
                final_idx[i] = du.correct_detection(valid_eta[:, i],
                                                    final_idx[i],
                                                    movement=2)
        # ----------------------------------------------------------------
        # 8) Return calculated time index into netCDF file
        # ----------------------------------------------------------------
        # Total number of POIs (valid AND invalid)
        spatiallen: int = len(lon)
        # original indices
        original_idx = np.arange(spatiallen)
        # Indices of the valid POIs
        val_idx = np.setdiff1d(original_idx, inv_idx)
        # Fill the each POI with information of arrival (if valid)
        # initialize array
        index_out = np.zeros_like(original_idx)
        # Fill valid POIs with arrival index
        index_out[val_idx] = final_idx
        # Non valid POIs are given a default value
        index_out[idx in inv_idx] = timelen-1
        # write output
        hio.write_output(file, index_out)


if __name__ == '__main__':
    main()
