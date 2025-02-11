import numpy as np
import scipy as sp
from pathlib import Path
import netCDF4 as nc
from HySEAio import read_outputs, get_fault_attributes, detect_invalid_values

"""
Functions for computing sta/lta and obtaining the trigger inidices
"""


def sta_lta(eta: np.ndarray, nsta: int, nlta: int, tolerance: float = 1e-10) -> np.ndarray:
    # The cumulative sum can be exploited to calculate a moving average (the
    # cumsum function is quite efficient)
    sta = np.cumsum(eta ** 2, dtype=np.float64)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta /= nsta
    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    # Pad zeros
    sta[:nlta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    # dtiny = np.finfo(0.0).smallest_normal
    dtiny = tolerance
    idxl = lta < dtiny
    lta[idxl] = dtiny
    stalta = sta/lta
    return sta, lta, stalta


def trigger(stalta: np.ndarray, threshold: float = 10):
    """
    Computes a boolean array indicating whether a tsunami signature
    is detected or not. Compares the sta/lta with a trigger value
    Parameters:
    stalta: np.ndarray. sta/lta computed earlier
    trigger: float. Threshold value
    Returns:
    detected. boolean array. False if not detected and True if yes
    first_detected. int. Index of the first trigger
    """
    # initialize boolean array
    detected = np.zeros_like(stalta, dtype=bool)
    # detect where the threshold is surpassed
    idx_detection = stalta > threshold
    detected[idx_detection] = True
    # if the threshold is not exceeded, no trigger is ensued
    if len(np.where(idx_detection)[0]) == 0:
        first_detected = 0
    else:
        first_detected = np.where(detected)[0][0]
    return detected, first_detected


# Gets the bigger detection idx between the sta/lta and the estimated
# arrival index
def get_bigger_idx(detected_idx, estimated_idx):
    """
    Returns an array conatining the bigger index between the sta/lta 
    (detected_idx) and the estimated index with the travel time 
    (estimated_idx)
    Parameters:
    detected_idx: np.ndarray. Array of sta/lta trigger indices
    estimated_idx: np.ndarray. Array of estimated travel time indices
    Returns:
    final_idx: np.ndarray. Bigger index between the past two
    """
    final_idx = [max(a, b) for a, b in zip(detected_idx, estimated_idx)]
    return final_idx

# Gets the better detection idx between the sta/lta and the estimated
# arrival index


def get_better_estimate(detected_idx, estimated_idx, detection_status):
    """
    Retunrs an array conatining the better estimation arrivale index 
    between the sta/lta (detected_idx) and the estimated index with 
    the travel time (estimated_idx) taking into account the correctness
    of the sta/lta detection and the bigger index between the two.
    If the sta/lta detection id correct, it is defined as the better 
    estimate. Otherwise, the bigger index is considered the better estimate
    Parameters:
    detected_idx: np.ndarray. Array of sta/lta trigger indices
    estimated_idx: np.ndarray. Array of estimated travel time indices
    detection_status: np.ndarray. Array of the correctnes of the estimation
    computed with check_arrival_detection
    Returns:
    final_idx: np.ndarray. Bigger index between the past two
    """
    # initialize the array with the better estimates
    final_idx = np.zeros_like(detected_idx)
    for i, (d, e) in enumerate(zip(detected_idx, estimated_idx)):
        if detection_status[i, :].all():
            final_idx[i] = d
        else:
            # legacy
            #final_idx[i] = max(d, e)
            # new experimental version
            final_idx[i] = min(d, e)
    return final_idx


# Check that the detection is "correct". That is, that the detected time
# is surrounded by 0 values


def check_arrival_detection(eta, i: int, n: int = 3, tol: float = 1e-4) -> bool:
    """
    Check that the dtection was succesful. A succesful detection is such that
    the index is before the arrival of a wave, that is, its value is zero and
    is surrounded by 0.
    Parameters:
    eta: np.ndarray. Elevation time series
    i: int. Index of detection
    n: int. Number of elements to check if detection was succesful or not.
    Returns:
    bool. False if it wasn't succesful, True otherwise
    """
    # Verify that there are sufficient number of elements at either side and
    # that there was a detection (if not, i==0)
    if (i == 0) or (i - n < 0) or (i + n >= len(eta)):
        return False
    # elements before i
    before = eta[i-n:i]
    # elements after i
    after = eta[i+1:i+n+1]
    # compare
    return abs(after-before) < tol

# TODO: add support for boolean elements in addtion to strings
# Function for correcting the detection if deemed incorrect by check_arrival_detection


def correct_detection(eta, i: int, movement: int = 1, n: int = 3, tol: float = 1e-4) -> int:
    """
    If the detection is unsuccessful (i.e. the detection occurs after the 
    arrival of the wave), the index is moved backwards until it is correct
    Parameters:
    eta: np.ndarray. Elevation time series
    i: int. Index of detection
    n: int. Number of elements to check if detection was succesful or not.
    """
    if (i <= movement) or (i+n >= len(eta)):
        return i
    else:
        # Checks the arrival
        check = check_arrival_detection(eta, i, n=n, tol=tol)
        j = i
        # new version
        while not check.all():
            # move the index backwards
            j -= movement
            if j <= movement:
                break
            else:
                check = check_arrival_detection(eta, j, n=n, tol=tol)
                if not hasattr(check, 'all'):
                    break
        return j


# crop time series. Strips leading zeros before wave arrival
def crop_time_series(eta: np.ndarray, idx: int, include: bool = True) -> np.ndarray:
    """
    Crops the input time series eta to by eliminating all elements before
    idx. If include is True, the idx-th element will be kept, otherwise
    it is removed
    Parameters:
    eta: np.ndarray. Elevation time series to crop
    idx:. int. Index of the detected wave arrival
    include. bool. Flag for including or eliminating arrival elevation
    point
    Returns:
    eta_cropped: np.ndarray. eta stripped of elements prior to idx
    """
    if include:
        return eta[idx:]
    else:
        return eta[idx+1:]


def reconstruct_cropped_time_series(eta_cropped: np.ndarray, original_len: int) -> np.ndarray:
    """
    Reconstruct the input cropped time series by adding the trailing 
    array of zeros that was cropped when saving to pickle
    Parameters:
    eta_cropped: np.ndarray: cropped elevation time series
    original_len: int: length of the original time series
    Returns:
    eta_recontructed: np.ndarray: reconstructed time series with trailing zeros
    """
    # length of the cropped time series
    cropped_len = len(eta_cropped)
    # difference between original and cropped time series, the amount of zeros to add
    len_diff = original_len - cropped_len
    # reconstruct
    eta_reconstructed = np.pad(
        eta_cropped, (len_diff, 0), 'constant', constant_values=0)
    return eta_reconstructed
