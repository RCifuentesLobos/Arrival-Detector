import numpy as np
import scipy as sp
from pathlib import Path
import greatcircle as grc
import os
import netCDF4 as nc
import glob
import pickle
from typing import Dict, Any, Tuple, List


# List data files
def list_outputs(directory: str, 
                 extension: str='.nc') -> List[str]:
    """
    List all files with extension in directory
    """
    # Extension needs to start with a point
    if not extension.startswith('.'):
        extension = '.' + extension
    # create patern to look for
    patern = os.path.join(directory, f'*{extension}')
    # lists all output files
    files = glob.glob(patern)
    return files


# List data files from all subdirectories in a directory
def list_outputs_recursively(directory: str, 
                             name: str='out_ts.nc') -> List[str]:
    """
    List all output files of names name in directories recursively
    """
    output_list = []
    for path, _, files in os.walk(directory):
        for file in files:
            if file == name:
                output_list.append(os.path.join(path, file))
    return output_list


# Read data files
def read_outputs(output: str):
    """
    Reads the HySEA output data
    Parameters:
    ----------
    output : str, 
            HySEA output file
    Returns:
    -------
    time : np.ndarray, 
        times at which the data is saved
    lon : np.ndarray, 
        longitudes of the POIs
    lat : np.ndarray, 
        latitudes of the POIs
    eta : np.ndarray, 
        Elevation at the POIs 
    mask_*: np.ndarray, 
        boolean array of maks values for * array
    fault_info: dict, 
        Dictionary with information about the fault
    """
    data = nc.Dataset(output, 'r')
    # get global attributes
    global_attributes = data.__dict__
    # number of faults
    nfaults = global_attributes['num_faults']
    # get data from faults
    fault_info = {f'fault_{k+1}': global_attributes[f'fault_{k+1}']
                  for k in range(nfaults)}
    # get time vector (it is a masked array)
    time = data.variables['time'][:]
    if np.ma.is_masked(time):
        mask_time = np.ma.getmask(time)
    else:
        mask_time = False

    # Get longitude and latitude of POIs
    lon = data.variables['longitude'][:]
    # check mask for longitude
    if np.ma.is_masked(lon):
        mask_lon = np.ma.getmask(lon)
    else:
        mask_lon = False

    # get latitude
    lat = data.variables['latitude'][:]
    # check mask for latitude
    if np.ma.is_masked(lat):
        mask_lat = np.ma.getmask(lat)
    else:
        mask_lat = False

    # get elevation
    eta = data.variables['eta'][:]
    if np.ma.is_masked(eta):
        mask_eta = np.ma.getmask(eta)
    else:
        mask_eta = False
    
    # get velocities
    ux = data.variables['ux'][:]
    uy = data.variables['uy'][:]
    if np.ma.is_masked(ux):
        mask_ux = np.ma.getmask(ux)
    else:
        mask_ux = False
    if np.ma.is_masked(uy):
        mask_uy = np.ma.getmask(uy)
    else:
        mask_uy = False

    return time, lon, lat, eta, ux, uy, \
           mask_time, mask_lon, mask_lat, \
           mask_eta, mask_ux, mask_uy, fault_info


# read depth of POI locations
def read_depth(depth_file: str) -> np.ndarray:
    """
    Reads the (lon,lat,depth) coordinates of POI locations
    Parameters:
    ----------
    depth_file : str, 
            POI locations
    Returns:
    -------
    depth : np.ndarray, 
        depth of the POIs in meters
    """
    locs = np.genfromtxt(depth_file)
    # get depth of the specified fault
    depth = locs[:, 2]
    return depth

# get indices of POIs with depth greater than coast_value (i.e. on land)
def get_onland_indices(depth_file: str,
                       coast_value: float = -1) -> np.ndarray:
    """
    Returns the indices of the POIs with depth greater than 0
    Parameters:
    ----------
    depth_file : str, 
        File containing POI locations (lon,lat,depth) coords
    coast_value : float,
        Depth value to consider as coast (default -1 m)
    Returns:
    -------
    onland_indices : np.ndarray, 
        Indices of the POIs on land
    """
    # read depth of POI locations
    depth = read_depth(depth_file)
    # boolean array of POIs on land
    onland = depth > coast_value
    # get indices of POIs with depth greater than coast_value
    onland_indices = np.nonzero(onland)[0]
    return onland_indices

# Get indices of POIs with distances smaller than min_distance. All POIs
#  with distances smaller than min_distance, won't be considered in the
#  cropping of the time series
def get_poi_idx_within_min_distance(flat: np.ndarray, flon: np.ndarray,
                                    lat: np.ndarray, lon: np.ndarray,                                  
                                    min_distance: float) -> np.ndarray:
    """
    Returns the indices of the POIs with distances smalle than min_distance
    from the fault. These POIs are not considered in the cropping
    Parameters:
    ----------
    flat : np.ndarray, 
            Fault latitudes
    flon : np.ndarray, 
        Fault longitudes
    lat : np.ndarray, 
        POI latitudes
    lon : np.ndarray, 
        POI longitudes
    min_distance : float, 
        Minimum distance from POI to fault to consider
    Returns:
    -------
    further_than_min_distance_idx : np.ndarray, 
        Indices of the POIs further than min_distance from the fault
    """
    # transform flat to ndarray
    flat = np.array(flat)
    flon = np.array(flon)
    # Get number of POIs
    npois = len(lat)
    # get number of faults, if there's more than one
    if flat.ndim > 1:
        nfaults = len(flat)
    else:
        nfaults = 1
    # initialize array to store distances and compute distances
    # case where there's only one fault
    if nfaults == 1:
        # initialize array to store distances
        dists_degrees = np.zeros(npois)
        # initialize array to store indices
        further_than_min_distance_idx = np.zeros(npois, dtype=np.bool_)
        for i in range(npois):
            dists_degrees[i] = grc.distance_in_degrees(
                lat[i], lon[i], flat, flon)
            # check if the distance is less than the min distance
            if dists_degrees[i] < min_distance:
                further_than_min_distance_idx[i] = True
    # case where there's more than one fault
    else:
        # initialize array to store distances
        dists_degrees = np.zeros((npois, nfaults))
        # initialize array to store indices
        further_than_min_distance_idx = np.zeros((npois,nfaults), dtype=np.bool_)
        for f in range(nfaults):
            for i in range(npois):
                dists_degrees[i, f] = grc.distance_in_degrees(
                    lat[i], lon[i], flat[f], flon[f])
                    # check if the distance is less than the min distance
                if dists_degrees[i, f] < min_distance:
                    further_than_min_distance_idx[i,f] = True
        # if at least one fault is closer than min_distance, then discard 
        # the POI for cropping (set to True)
        further_than_min_distance_idx = np.any(further_than_min_distance_idx, axis=1)
    return further_than_min_distance_idx


# Get indices of POIs with distances bigger than min_distance. All POIs
# with distances bigger than min_distance, won't be considered in the
# cropping of the time series
def get_poi_idx_outside_min_distance(flat: np.ndarray, flon: np.ndarray,
                                    lat: np.ndarray, lon: np.ndarray,                                  
                                    min_distance: float) -> np.ndarray:
    """
    Returns the indices of the POIs with distances bigger than min_distance
    from the fault. These POIs are not considered in the cropping
    Parameters:
    ----------
    flat : np.ndarray, 
            Fault latitudes
    flon : np.ndarray, 
        Fault longitudes
    lat : np.ndarray, 
        POI latitudes
    lon : np.ndarray, 
        POI longitudes
    min_distance : float, 
        Minimum distance from POI to fault to consider
    Returns:
    -------
    within_than_min_distance_idx : np.ndarray, 
        Indices of the POIs further than min_distance from the fault
    """
    # transform flat to ndarray
    flat = np.array(flat)
    flon = np.array(flon)
    # Get number of POIs
    npois = len(lat)
    # get number of faults, if there's more than one
    if flat.ndim > 1:
        nfaults = len(flat)
    else:
        nfaults = 1
    # initialize array to store distances and compute distances
    # case where there's only one fault
    if nfaults == 1:
        # initialize array to store distances
        dists_degrees = np.zeros(npois)
        # initialize array to store indices
        less_than_min_distance_idx = np.zeros(npois, dtype=np.bool_)
        for i in range(npois):
            dists_degrees[i] = grc.distance_in_degrees(
                lat[i], lon[i], flat, flon)
            # check if the distance is bigger than the min distance
            if dists_degrees[i] > min_distance:
                less_than_min_distance_idx[i] = True
    # case where there's more than one fault
    else:
        # initialize array to store distances
        dists_degrees = np.zeros((npois, nfaults))
        # initialize array to store indices
        less_than_min_distance_idx = np.zeros((npois,nfaults), dtype=np.bool_)
        for f in range(nfaults):
            for i in range(npois):
                dists_degrees[i, f] = grc.distance_in_degrees(
                    lat[i], lon[i], flat[f], flon[f])
                    # check if the distance is bigger than the min distance
                if dists_degrees[i, f] > min_distance:
                    less_than_min_distance_idx[i,f] = True
        # if at least one fault is further than min_distance, then discard 
        # the POI for cropping (set to True)
        less_than_min_distance_idx = np.any(less_than_min_distance_idx, axis=1)
    return less_than_min_distance_idx

# get the indices of POIs signals whose maximum amplitude is less than a 
# predefined threshold
def get_signal_amplitude_below_threshold(eta: np.ndarray,
                                         min_amplitude: float) -> np.ndarray:
    """
    Returns the indices of the POIs with maximum amplitude smaller
    than the threshold, effectively not taking into account POIs 
    with no registered signal
    Parameters:
    ----------
    eta : np.ndarray, 
        Elevation at the POIs
    min_amplitude : float, 
        Minimum amplitude threshold
    Returns:
    -------
    below_threshold_idx : np.ndarray,
        array of POI indices whose max amplitudes are below min_amplitude
    """
    # get number of POIs (rows are time, columns are POIs)
    npois = eta.shape[1]
    # initialize array to store indices
    below_threshold_idx = np.zeros(npois, dtype=np.bool_)
    # loop through POIs
    for i in range(npois):
        # check if the maximum amplitude is below the threshold 
        if np.max(np.abs(eta[:, i])) < min_amplitude:
            below_threshold_idx[i] = True
    return below_threshold_idx

# get the indices of POIs signals whose maximum elevation is less than 
# variable threshold function of its depth. 
# This threshold is defined by the helper function:
# a)  _get_poi_threshold

def _get_poi_threshold(depth_file: str,
                       minamp: float,
                       coast_value: float | None = None) -> np.ndarray:
    """
    Reads the POI depths and returns the variable threshold
    Parameters:
    -----------
    depth_file : str, 
        File containing POI locations (lon,lat,depth) coords
    minamp : float,
        Minimum amplitude constant for the threshold function.
    coast_value : float | None,
        Depth value (in meters) to consider as coast. If None, coast defaults to 0 m
    Returns:
    --------
    threshold : np.ndarray, 
        threshold array for every POI
    """
    # read depth of POI locations
    depth = read_depth(depth_file)
    # apply threshold function only to ocean POIs (depth < coast_value)
    if coast_value is None:
        coast_value = 0
    # disregard land POIs (depth >= coast_value). 
    # They will always be filtered out
    depth[depth >= coast_value] = np.nan
    # compute threshold as a function of depth
    threshold = minamp / (np.abs(depth)**0.25)
    return threshold

# b) then, the main filter function uses this helper function
def get_signal_amplitude_below_greenslaw(eta: np.ndarray,
                                         depth_file: str,
                                         minamp: float,
                                         coast_value: float | None = None) -> np.ndarray:
    """
    Returns the indices of the POIs with maximum amplitude smaller
    than the threshold given by Green's Law 
    Meant to replace get_signal_amplitude_below_threshold
    Parameters:
    ----------
    eta : np.ndarray, 
        Elevation at the POIs
    depth_file : str, 
        File containing POI locations (lon,lat,depth) coords
    minamp : float,
        Minimum amplitude constant for the threshold function.
    coast_value : float | None,
        Depth value (in meters) to consider as coast. If None, coast defaults to 0 m
    Returns:
    -------
    below_threshold_idx : np.ndarray,
        array of POI indices whose max amplitudes are below the threshold given by Green's Law
    """
    # get number of POIs (rows are time, columns are POIs)
    npois = eta.shape[1]
    # initialize array to store indices
    below_threshold_idx = np.zeros(npois, dtype=np.bool_)
    # get threshold array
    threshold = _get_poi_threshold(depth_file, minamp=minamp, coast_value=coast_value)
    # loop through POIs
    for i in range(npois):
        # check if the maximum amplitude is below the threshold 
        if np.max(np.abs(eta[:, i])) < threshold[i]:
            below_threshold_idx[i] = True
    return below_threshold_idx


# Create dictionaries of the attributes of the faults from global data attributes
def get_fault_attributes(fault_info: dict) -> list:
    """
    Creates dictionaries with data from the faults used in the simulation
    Parameters:
    ----------
    fault_info : dict, 
        A dictionary with a key of the form 'fault_n' and values
        in the format:
            time: 0.00, 
            lon_barycenter: 288.92, 
            lat_barycenter: -19.43, 
            rake: 90.00, 
            slip: 10.00
    Returns: 
    -------
    fault_attributes_list : list[dict], 
        Dictionary with keys:
            time
            lon_barycenter
            lat_barycenter
            rake
            slip
    """
    # initialize list
    fault_attributes_list = []
    # loop though faults
    for k in fault_info.keys():
        # obtain a lista of key:value pairs for every entry
        aux = fault_info[k].split(',')
        fault_attributes_list.append(
            {aux[i].split(':')[0].lstrip(): float(aux[i].split(':')[1].lstrip()) for i in range(len(aux))})
    return fault_attributes_list


# Detect and return only valid POI elevation time series
def detect_invalid_values(mask: np.ndarray, 
                          eta: np.ndarray) -> Tuple[np.ndarray,
                                                    np.ndarray]:
    """
    Detects the invalid maksed elements. Elevation data eta has
    dimensions (time, nPOIs). Invalid data is located in some 
    column defined by the column of True values of a mask. The 
    index of this(these) column(s) is the index of the POI with
    invalid values
    Parameters:
    ----------
    mask : np.ndarray, 
        Mask of eta
    eta : np.ndarray, 
        Elevation array
    Returns:
    -------
    inv_idx : np.ndarray, 
        Indices of invalid values
    masked_eta : np.ndarray, 
        Elevation matrix without invalid data
    """
    # indices of the column of True values
    inv_idx = np.where(mask.any(axis=0))[0]
    masked_eta = np.delete(eta, inv_idx, axis=1)
    return inv_idx, masked_eta


# Returns the latitude and longitude of the valid POIs
def get_valid_coordinates(lon: float, 
                          lat: float, 
                          idx: int) -> Tuple[np.ndarray,
                                             np.ndarray]:
    """
    Returns the coordinates where there are valid data entries

    """
    valid_lon = np.delete(lon, idx)
    valid_lat = np.delete(lat, idx)
    return valid_lon, valid_lat


# Saves the calculated index to a nc file
def write_output(output: str, 
                 index_out: np.ndarray) -> None:
    """
    Saves the calculated arrival indices to an output nc file
    Parameters:
    ----------
    output : str, 
        Name of the output nc file
    index_out : np.ndarray, 
        Array of the indices
    """
    data_write = nc.Dataset(output, 'a')
    time_idx = data_write.createVariable(
        'start_index', 'i2', ('grid_npoints',))
    time_idx[:] = index_out
    data_write.close()


# load pickle data
def load_pickle_elevation(filename: str, 
                          ret_all: bool = False):
    """
    Loads the pickle file data and rebuilds the time series
    Parameters:
    ----------
    filename: str,
        Name of the pickle file
    ret_all: bool,
        Flag for returning the whole data dict or not.
        Defaults to False, not returning the whole dictionary
    """
    # load data
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    # 0) retrieve valid ids and all valid pois in boolean dtype
    valid_ids = data['valid_ids']
    all_ids = data['all_valid_pois']
    # rebuild data and time series
    # 1) rebuild time array
    # initial time
    ti = data['start_time']
    # final time
    tf = data['end_time']
    # length of time array
    ntimes = data['ntimes']
    # create array
    time = np.linspace(ti, tf, ntimes)
    # 2) rebuild time series and save max values and times
    # initialize array
    # number of time series
    npois = data['npois']
    # initialize time series array
    # for elevation
    eta = np.zeros((ntimes, npois))
    # for velocity u_x
    ux = np.zeros((ntimes, npois))
    # for velocity u_y
    uy = np.zeros((ntimes, npois))
    # initialize array for cropped times
    crop_times = np.zeros(npois)
    # initialize array for maximum elevation
    max_elev = np.zeros(npois)
    # initialize array for maximum amplitude
    max_amp = np.zeros(npois)
    # initialize array for time of arrival of max elevation
    time_max_elev = np.zeros(npois)
    # initialize array for time of arrival of max amplitude
    time_max_amp = np.zeros(npois)
    # loop though all cropped time series
    for i, id in enumerate(valid_ids):
        # 1) retrieve the cropped time series
        eta_crop = data[id] # elevation time series
        ux_crop = data[f'ux_{id}'] # velocity u_x time series
        uy_crop = data[f'uy_{id}'] # velocity u_y time series
        # 1.1) length of cropped time series
        lec = len(eta_crop)
        # 1.2) assign the last lec elements
        eta[-lec:, i] = eta_crop
        ux[-lec:, i] = ux_crop
        uy[-lec:, i] = uy_crop
        # 2) retrieve the time of crop
        crop_times[i] = data[f'crop_time_{id}']
        # 3) retrieve maximum elevation
        max_elev[i] = data[f'max_tsunami_elev_{id}']
        # 4) retrieve maximum amplitude
        max_amp[i] = data[f'max_tsunami_amp_{id}']
        # 5) retrieve time of arrival of max elevation
        time_max_elev[i] = data[f'max_tsunami_elev_time_{id}']
        # 6) retrieve time of arrival of max amplitude
        time_max_amp[i] = data[f'max_tsunami_amp_time_{id}']
    # 3) retrieve coordinates
    lon = data['lon']
    lat = data['lat']
    # if ret_all is True returns everything
    if ret_all:
        return data, time, crop_times, lon, lat, eta, ux, uy,\
            max_amp, max_elev, time_max_amp, time_max_elev, valid_ids, all_ids
    else:
        return time, crop_times, lon, lat, eta, ux, uy


# trasnform pickle to netcdf
def pickle2nc(filename: str,
              ncfilename: str | None = None) -> None:
    """
    Transforms a pickle file into a netcdf format
    Parameters:
    ----------
    filename: str,
        Name of the pickle file
    ncfilename: str | None,
        Name of the output netcdf file. If None, it will be named
        as <pickle_filename>_post.nc
    """
    # 1) load pickle data
    data, time, crop_time, lon, lat, eta, ux, uy,\
        max_amp, max_elev, \
        time_max_amp, time_max_elev, \
        valid_ids, all_ids = load_pickle_elevation(
                             filename, ret_all=True)
    npois = data['npois']
    # 2) create nc file
    # 2.1) nc file name
    if ncfilename is None:
        ncfilename = f"{filename.split('.')[0]}_post.nc"
    nc_file = nc.Dataset(ncfilename, 'w', format='NETCDF4')
    # 3) create dimensions
    # 3.1) for time
    timedim = nc_file.createDimension('time', len(time))
    # 3.2) for coordinates
    londim = nc_file.createDimension('lons', len(lon))
    latdim = nc_file.createDimension('lats', len(lat))
    # 3.3) for time series
    npoidim = nc_file.createDimension('npois', npois)
    # 3.4) for crop timestamps
    croptimedim = nc_file.createDimension('crop_time', len(crop_time))
    # 3.5) for max amp and max elevations
    maxvaluedime = nc_file.createDimension('max_tsunami', len(max_amp))
    # 3.6) for arrival times of amplitude and elevations
    timemaxdime = nc_file.createDimension('timemax', len(time_max_amp))
    # 3.7) for all valid identifiers of POIs
    valididdim = nc_file.createDimension('valid_id', len(valid_ids))
    # 3.8) for all boolean POI indices of validity
    alliddim = nc_file.createDimension('all_id', len(all_ids))
    # 4) create variables
    # 4.1) for time
    time_var = nc_file.createVariable('time', np.float32, ('time',))
    # 4.2) for coordinates
    lon_var = nc_file.createVariable('lon', np.float32, ('lons',))
    lat_var = nc_file.createVariable('lat', np.float32, ('lats',))
    # 4.3) for time series
    # 4.3.3) for elevation
    eta_var = nc_file.createVariable('eta', np.float32, ('time', 'npois'))
    # 4.3.4) for velocity u_x
    ux_var = nc_file.createVariable('ux', np.float32, ('time', 'npois'))
    # 4.3.5) for velocity u_y
    uy_var = nc_file.createVariable('uy', np.float32, ('time', 'npois'))
    # 4.4) for crop timestamps
    crop_time_var = nc_file.createVariable(
        'crop_time', np.float32, ('crop_time',))
    # 4.5) for max amp and max elevations
    max_amp_var = nc_file.createVariable(
        'max_amplitude', np.float32, ('npois',))
    max_elev_var = nc_file.createVariable(
        'max_elevation', np.float32, ('npois',))
    # 4.6) for arrival times of amplitude and elevations
    max_amp_time_var = nc_file.createVariable(
        'max_tsunami_amplitude_time', np.float32, ('npois',))
    max_elev_time_var = nc_file.createVariable(
        'max_tsunami_elevation_time', np.float32, ('npois',))
    # 4.7) for all valid identifiers of POIs
    valid_ids_var = nc_file.createVariable('valid_ids', np.int32, ('npois',))
    # 4.8) for all indices of validity for POIs
    all_ids_var = nc_file.createVariable('all_ids', 'i1', ('all_id',))
    all_ids_var.units = 'boolean (1=True, 0=False)'
    # 5) assign values to variable
    # 5.1) for time
    time_var[:] = time
    # 5.2) for coordinates
    lon_var[:] = lon
    lat_var[:] = lat
    # 5.3) for time series
    # 5.3.1) for elevation
    eta_var[:, :] = eta
    # 5.3.2) for velocity u_x
    ux_var[:, :] = ux
    # 5.3.3) for velocity u_y
    uy_var[:, :] = uy
    # 5.4) for crop timestamps
    crop_time_var[:] = crop_time
    # 5.5) for max amp and max elevations
    max_amp_var[:] = max_amp
    max_elev_var[:] = max_elev
    # 5.6) for arrival times of amplitude and elevations
    max_amp_time_var[:] = time_max_amp
    max_elev_time_var[:] = time_max_elev
    # 5.7) for all valid identifiers of POIs
    valid_ids_var[:] = valid_ids
    # 5.8) for all indices
    all_ids_var[:] = all_ids.astype('i1')
    # close nc file
    nc_file.close()
