import HySEAio as hio
import detectutils as du
import POIs as poi
import greatcircle as grc
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
import os

# ------------------------------------------------------------------------
# A) Define directory structure
# ------------------------------------------------------------------------
# directory with output files
#dir_out: str = "/Users/rodrigocifuentes/Documents/Alemania/Universidad/Colaboraciones/INGV/Codes/tested_codes/Testdata/alaska"
dir_out: str = "/Users/rodrigocifuentes/Documents/Alemania/Universidad/Colaboraciones/INGV/Codes/tested_codes/arrival_detector"
# file containing depth data
#depth_file: str = "/Users/rodrigocifuentes/Documents/Alemania/Universidad/Colaboraciones/INGV/Codes/tested_codes/Testdata/alaska/ts_pac_0-360_wdepth.dat"
depth_file: str = "/Users/rodrigocifuentes/Documents/Alemania/Universidad/Colaboraciones/INGV/Codes/tested_codes/arrival_detector/ts_pac_first_depth.dat"
# ------------------------------------------------------------------------
# B) Define STA/LTA parameters
# ------------------------------------------------------------------------
# time interval in seconds between elevation computation
deltat: int = 30
# define sta/lta and detection parameters
sta_window: int = 20
lta_window: int = 600
trigger_value: int = 8

# ------------------------------------------------------------------------
# C) define POI discrination criteria
# ------------------------------------------------------------------------
# minimum distance from POI to fault to consider for arrival algorithm
min_distance: float = 25 # in degrees

# ------------------------------------------------------------------------
# D) Define which filters to apply
# ------------------------------------------------------------------------
# Filter by depth? If True, skip POIs located on land
filter_depth: bool = False
# depth value (in meters) to consider as coast
coast_value: float = -1 
# Filter by distance? If True, apply distance filter before processing
# This filters makes that all time series of POIs within certain range
# are NOT cropped. This is done by setting the arrival time at 0
filter_distance: bool = False
# Filter by amplitude? If True, apply amplitude filter before processing
filter_amplitude: bool = True

# ------------------------------------------------------------------------
# E) set verbosity level
# ------------------------------------------------------------------------
verbose: bool = True

def main(dir_out: str = dir_out,
         deltat: float = deltat,
         sta_window: int = sta_window,
         lta_window: int = lta_window,
         trigger_value: int = trigger_value,
         depth_file: str = depth_file,
         filter_depth: bool = filter_depth,
         coast_value: float | None = coast_value,
         filter_distance: bool = filter_distance,
         min_distance: float = min_distance,
         filter_amplitude: bool = filter_amplitude,
         verbose: bool = verbose) -> None:
    """
    Parameters:
    -----------
    dir_out : str, 
        The directory with .nc outputs
    deltat : float, 
        The time interval in seconds between succesive
        elevation data points
    sta_window : int, 
        The length of the STA window
    lta_window : int, 
        The length of the LTA window
    trigger_value : int, 
        Threshold for detecting an event
    depth_file : str, 
        File with depth data
    filter_depth : bool, 
        If True, skip POIs located on land
    coast_value : float | None,
        Depth value (in meters) to consider as coast
    distance_filter : bool, 
        If True, apply distance filter before processing
    min_distance : float, 
        Minimum distance from POI to fault to consider
    filter_amplitude : bool,
        If True, apply amplitude filter before processing
    verbose : bool, 
        If True, print information during processing
    """
    # list all output files
    files = hio.list_outputs_recursively(dir_out)
    # loop through all output files
    for file in files:
        print(f'[Info] Processing {file}')
        # ----------------------------------------------------------------
        # 1) Read data and apply filters
        # ----------------------------------------------------------------
        # read data
        time, lon, lat, \
            eta, _, _, _, mask_eta, \
            fault_info = hio.read_outputs(file)
        # get total number of POIs
        total_pois = np.shape(eta)[1]
        # get original amount of invalid POIs
        n_invalid_pois = np.sum(np.any(mask_eta, axis=0))
        if verbose:
            print(f"[Info] Total number of POIs: {total_pois}")
            print(f"[Info] Number of invalid POIs before filtering: {n_invalid_pois}")
        # If wanted, skip POIs located on land. All POIs with depth 
        # larger than 0 (on land) are masked (won't be considered
        # in the cropping process)
        if filter_depth:
            if verbose:
                print(f"[Filter] Applying depth filter:\n", 
                      f"Skipping POIs located on land. Coastline depth threshold: {coast_value} (m)")
            if coast_value is None:
                coast_value = 0
            onland_indices = hio.get_onland_indices(depth_file,
                                                    coast_value=coast_value)
            # modify mask. If i-th column is True, that POI is masked
            mask_eta[:, onland_indices] = True
            if verbose:
                print(f"[Filter] Filtered {len(onland_indices)} POIs located on land")
        # get fault information
        fault_info_list = hio.get_fault_attributes(fault_info)
        # fault lon and lat
        flon: float = fault_info_list[0]['lon_barycenter']
        flat: float = fault_info_list[0]['lat_barycenter']
        # If wanted, apply distance (fault<->POI) filter before 
        # processing. All POIs with distances smaller than min_distance
        # won't be cropped
        # initialize boolean array to identify those POIs within 
        # min_distance that won't be cropped
        is_poi_within = np.zeros((total_pois), dtype=np.bool_)
        if filter_distance:
            if verbose:
                print(f"[Filter] Applying distance filter\n:",
                      f" skipping POIs within {min_distance} degrees from fault")
            # get indices of POIs within min_distance from a fault
            within_poi_idx = hio.get_poi_idx_within_min_distance(flat, flon, 
                                                                 lat, lon,
                                                                 min_distance)
            # True if POI is within range of min_distance
            is_poi_within[within_poi_idx] = True
            if verbose:
                print(f"[Filter] Filtered {len(within_poi_idx)}", 
                      f"POIs within {min_distance} degrees from fault")
        # If wanted, apply amplitude (signal amplitude) filter before 
        # cropping. All POIs' signal with a maximum amplitude smaller
        # than val/(depth)**0.25 are masked (wont' be considered in the 
        # cropping). val is by default 0.05 (see helper function 
        # _get_poi_threshold() in HySEAio.py)
        if filter_amplitude:
            if verbose:
                print(f"[Filter] Applying amplitude filter")
            no_amplitude_indices = hio.get_signal_amplitude_below_greenslaw(eta,
                                                                           depth_file,
                                                                           coast_value=coast_value)
            # modify mask. If i-th column is True, that POI is masked
            mask_eta[:, no_amplitude_indices] = True
            if verbose:
                print(f"[Filter] Filtered {len(np.argwhere(no_amplitude_indices))}",
                      "POIs with amplitude below threshold given by Green's Law")
        # get all ids (starting from zero) of valid and invalid POIs
        # note: invalid POIs are masked as True in mask
        all_ids = np.any(mask_eta, axis=0)
        # print invalid pois
        if verbose:
            n_inv_pois = np.sum(all_ids)
            print(f"[Info] Total number of invalid POIs not considered: {n_inv_pois}")
        # get only valid data (according to mask)
        # valid_eta has dimensions (timelen, Total # POIs - masked POIs) 
        inv_idx, valid_eta = hio.detect_invalid_values(mask_eta, eta)
        # Get length of time series and number of valid POIs
        timelen: int = len(time)
        # number of valid POIs (without True values in mask_eta)
        npois: int = np.shape(valid_eta)[1]
        if verbose:
            print(f"[Info] Number of valid POIs considered: {npois}") 
        # get id for each individual POI.
        # note this starts from 1, while indices from inv_idx start from 0.
        identifiers = np.arange(1, npois + len(inv_idx) + 1)
        # +1 to inv_idx to convert indices from 0 - npois to 1 - npois+1
        # e.g. element 1720 is masked in mask_eta:
        # in identifiers this will correspond to 1721. This idx will not
        # be in valid_ids, so: 
        # valid_ids[1718:1722] = array([1719, 1720, 1722, 1723])
        valid_ids = identifiers[~np.isin(identifiers, inv_idx+1)]
        valid_ids = valid_ids[:npois]
        # get valid coordinates
        vlon, vlat = hio.get_valid_coordinates(lon, lat, inv_idx)
            

        # ----------------------------------------------------------------
        # 2) Compute STA/LTA for all POIs with 
        # a) depth < coast_value, 
        # b) distance > min_distance
        # c) amplitude > threshold
        # ----------------------------------------------------------------
        # initialize array with the detection values
        detection_idx = np.zeros(npois)
        # loop through all POIs
        for i in range(npois):
            # compute sta/lta
            _, _, stalta = du.sta_lta(valid_eta[:, i], sta_window, lta_window)
            # get detection values
            _, detection_idx[i] = du.trigger(stalta, threshold=trigger_value)
        # transform to integer
        detection_idx = [i.astype(int) for i in detection_idx]

        # ----------------------------------------------------------------
        # 3) Check if detection is correct
        # ----------------------------------------------------------------
        # initialize the boolean array
        nborders: int = 3
        detection_status = np.zeros((len(detection_idx), nborders), dtype=np.bool_)
        # loop through all detections
        for i, idx in enumerate(detection_idx):
            detection_status[i, :] = du.check_arrival_detection(valid_eta[:, i],
                                                                idx, n=nborders)

        # ----------------------------------------------------------------
        # 4) Compute the distance between fault and POI if STA/LTA is
        # incorrect
        # ----------------------------------------------------------------
        # initialize array with distances and travel times
        dist_tt = np.zeros((npois, 2))
        # loop through all POIs
        for i, (lo, la) in enumerate(zip(vlon, vlat)):
            # check if detection is not correct
            if not detection_status[i, :].all():
                dist_tt[i, 0] = grc.haversine(lo, la, flon, flat)
                dist_tt[i, 1] = grc.estimate_travel_time(lo, la, flon, flat)
        # Get index
        # time interval in seconds between elevation computation
        deltat: int = 30
        arrival_idx: int = np.floor(dist_tt[:, 1]/deltat).astype(int)

        # ----------------------------------------------------------------
        # 5) Check which value is better (the bigger, the more zeros are cropped)
        # ----------------------------------------------------------------
        final_idx = du.get_better_estimate(detection_idx, arrival_idx,
                                           detection_status)
        # ----------------------------------------------------------------
        # 6) Check if the detection was successful (only those that\
        #  weren't already)
        # ----------------------------------------------------------------
        # initialize the boolean array for the detection status after
        # checking the best estimate
        detection_status_after = np.zeros((len(final_idx),
                                           nborders), dtype=bool)
        # Loop through all detection status that weren't already correct
        for i, idx in enumerate(final_idx):
            if detection_status[i, :].all():
                detection_status_after[i, :] = detection_status[i, :]
            else:
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
        # ----------------------------------------------------------------
        # 8) Sets arrival index to 0 if POI is within min_distance
        # ----------------------------------------------------------------
        # For all valid_ids (note: they start at 1) it checks whether the
        # POI is within min_distance of any subfault. If it is, it sets 
        # the arrival index to zero. This guarantees that this time series
        # is not cropped
        for idx, val_ids in enumerate(valid_ids):
            if is_poi_within[val_ids - 1]:
                final_idx[idx] = 0

        # ----------------------------------------------------------------
        # 9) Crops the elevation time series eliminating the leading zeros
        # up until the arrival index. It saves the time series to a
        # dictionary to save it to a pickle file
        # ----------------------------------------------------------------
        # poopulate a dictionary to serialize the cropped elevation to
        # a .pkl file
        # initialize the dictionary
        cropped_time_series = {}
        # number of (valid) POIs
        cropped_time_series['npois'] = npois
        # validity of all POIs (True: valid, False: invalid)
        # note: starts from zero! as per the previous example:
        # if 1720 is invalid, the idx 1720 will be False
        # all_ids has to be bool
        if all_ids.dtype != np.bool_:
            all_ids = np.array(all_ids, dtype=bool)
        cropped_time_series['all_valid_pois'] = ~all_ids
        # save time data
        cropped_time_series['start_time'] = time[0]
        cropped_time_series['end_time'] = time[-1]
        cropped_time_series['ntimes'] = timelen
        # save coordinates
        cropped_time_series['lon'] = vlon
        cropped_time_series['lat'] = vlat
        # save fault information
        for fault in fault_info.keys():
            cropped_time_series[fault] = fault_info[fault]
        # save elevation information
        for i, idx in enumerate(final_idx):
            # save cropped time series
            cropped_time_series[valid_ids[i]] = du.crop_time_series(valid_eta[:, i],
                                                                    idx)
            # get index of maximum elevation (taking into account only positive)
            max_elevation_idx = np.argmax(valid_eta[:, i])
            # get index of maximum amplitude (taking into account entire amplitude)
            max_amplitude_idx = np.argmax(np.abs(valid_eta[:, i]))
            # save crop time for each poi time series
            cropped_time_series[f'crop_time_{valid_ids[i]}'] = time[idx]
            # save maximum elevation post cropping
            cropped_time_series[f'max_tsunami_elev_{valid_ids[i]}'] = valid_eta[max_elevation_idx, i]
            # save maximum amplitude post cropping
            cropped_time_series[f'max_tsunami_amp_{valid_ids[i]}'] = valid_eta[max_amplitude_idx, i]
            # save maximum elevation time
            cropped_time_series[f'max_tsunami_elev_time_{valid_ids[i]}'] = time[max_elevation_idx]
            # save maximum amplitude time
            cropped_time_series[f'max_tsunami_amp_time_{valid_ids[i]}'] = time[max_amplitude_idx]
        # save valid ids
        cropped_time_series['valid_ids'] = valid_ids
        # save to pickle
        pklfilename = f"{file.split('/')[-1].split('.')[0]}.pkl"
        pklpath = os.path.dirname(file)
        pklfilename = os.path.join(pklpath, pklfilename)
        with open(pklfilename, 'wb') as pklfile:
            pickle.dump(cropped_time_series, pklfile)
        if verbose:
            print(f'[Info] {file} done!')


if __name__ == '__main__':
    main()
