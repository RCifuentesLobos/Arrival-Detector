import HySEAio as hio
import detectutils as du
import POIs as poi
import greatcircle as grc
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
import os

# directory with output files
dir_out: str = "/Users/rodrigocifuentes/Documents/Alemania/Universidad/Colaboraciones/INGV/Codes/tested_codes/Testdata/alaska"
# file containing depth data
depth_file: str = "/Users/rodrigocifuentes/Documents/Alemania/Universidad/Colaboraciones/INGV/Codes/tested_codes/Testdata/alaska/ts_pac_0-360_wdepth.dat"
# time interval in seconds between elevation computation
deltat: int = 30
# define sta/lta and detection parameters
sta_window: int = 20
lta_window: int = 600
trigger_value: int = 8
# define POI discrination criteria
# minimum distance from POI to fault to consider for arrival algorithm
min_distance: float = 10 # in degrees
# minimum amplitude tolerance for POI to be considered (amplitude
# in meters within which the signal is considered noise, above the 
# threshold, it is considered a tsunami signal)
min_amplitude: float = 0.05 # in meters



def main(dir_out: str = dir_out,
         deltat: float = deltat,
         sta_window: int = sta_window,
         lta_window: int = lta_window,
         trigger_value: int = trigger_value,
         depth_file: str = None,
         filter_depth: bool = False,
         filter_distance: bool = True,
         min_distance: float = min_distance,
         amplitude_filter: bool = True,
         min_amplitude: float = min_amplitude) -> None:
    """
    Parameters:
    -----------
    dir_out: str. The directory with .nc outputs
    deltat: float. The time interval in seconds between succesive
      elevation data points
    sta_window: int. The length of the STA window
    lta_window: int. The length of the LTA window
    trigger_value: int. Threshold for detecting an event
    depth_file: str. File with depth data
    filter_depth: bool. If True, skip POIs located on land
    distance_filter: bool. If True, apply distance filter before processing
    min_distance: float. Minimum distance from POI to fault to consider
    amplitude_filter: bool. If True, apply amplitude filter before processing
    min_amplitude: float. Minimum amplitude tolerance for POI to be considered
    """
    # list all output files
    files = hio.list_outputs_recursively(dir_out)
    # loop through all output files
    for file in files:
        print(f'Processing {file}')
        # read data
        time, lon, lat, \
            eta, mask_time, mask_lon, mask_lat, mask_eta, \
            fault_info = hio.read_outputs(file)
        # If wanted, skip POIs located on land
        if filter_depth:
            onland_indices = hio.get_onland_indices(depth_file)
            # modify mask. If i-th column is True, that POI is masked
            mask_eta[:, onland_indices] = True
        # get only valid data (according to mask)
        inv_idx, valid_eta = hio.detect_invalid_values(mask_eta, eta)
        # Get length of time series and number of valid POIs
        timelen: int = len(time)
        npois: int = np.shape(valid_eta)[1]
        # get id for each individual POI.
        # note this starts from 1, while indices from inv_idx start from 0.
        identifiers = np.arange(1, npois + len(inv_idx) + 1)
        # +1 to inv_idx to convert indices from 0 - npois to 1 - npois+1
        valid_ids = identifiers[~np.isin(identifiers, inv_idx+1)]
        valid_ids = valid_ids[:npois]
        # get valid coordinates
        vlon, vlat = hio.get_valid_coordinates(lon, lat, inv_idx)
        # get fault information
        fault_info_list = hio.get_fault_attributes(fault_info)
        # fault lon and lat
        flon: float = fault_info_list[0]['lon_barycenter']
        flat: float = fault_info_list[0]['lat_barycenter']
        # If wanted, apply distance (fault<->POI) filter before 
        # processing. All POIs with distances smaller than min_distance
        # are masked (wont' be considered in the cropping process)
        if filter_distance:
            further_than_min_distance_idx = hio.get_poi_idx_within_min_distance(flat, flon, 
                                                                                lat, lon,
                                                                                min_distance)
            mask_eta[:, further_than_min_distance_idx] = True

        # ----------------------------------------------------------------
        # 2) Compute STA/LTA for all POIs with depth < 0
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
        detection_status = np.zeros((len(detection_idx), nborders), dtype=bool)
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
        # 8) Crops the elevation time series eliminating the leading zeros
        # up until the arrival index. It saves the time series to a
        # dictionary to save it to a pickle file
        # ----------------------------------------------------------------
        # poopulate a dictionary to serialize the cropped elevation to
        # a .pkl file
        # initialize the dictionary
        cropped_time_series = {}
        # number of POIs
        cropped_time_series['npois'] = npois
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
        print(f'{file} done!')


if __name__ == '__main__':
    main()
