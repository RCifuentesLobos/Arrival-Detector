
# Arrival_detector

Codes for detecting the wave arrival for .nc files outputs from HySEA

## Features

Arrival_detector contains codes for estimating the index of the wave arrival in an elevation time series and cropping the leading zero-elevation array, given by a HySEA simulation.  
It does this by combinating two approaches. An STA/LTA algorithm based on automatic phase picking algorithms used in seismology and a coarse travel time estimation using the long wave phase velocity and source-to-destination distance.  
It also contains codes for checking the detection and correcting it in case that both approaches fail.  

It loops through all .nc files in a given directory. If the main directory given as input contains more directories, it loops through all subdirectories looking for .nc files to be processed.  

It outputs new .nc files with only the crucial information about the simulation saved.

### File description

1. arrival_detector.py and arrival_detector_stalta.py
    - Main functions for looping through simulation outputs detecting arrivals and cropping elevation time series.
        - arrival_detector.py checks first the travel-time estimation and then corrects it with STA/LTA detection. It's slower than arrival_detector_stalta.py and not recommended.
        - arrival_detector_stalta.py performs an sta/lta wave picking and complements incorrectly detected waves with travel-time estimations detections. We recommend the use of this function.
2. HySEAio.py
    - Contains functions for reading and writing .nc files and pickle files
3. greatcircle.py
    - Contains functions for computing distances and calculating travel times.
4. POIs.py
    - Defines the POI class. Contains methods for visualizing the POIs contained in the input .nc file.
5. detectutils.py  
    - contains functions for the detection, checking and correction of wave arrivals, as well as cropping the time series

### How to use the main functions (arrival_detector_stalta.py is recommended)

- In arrival_detector_stalta.py:
    1. Define the directory with the outputs in dir_out
    2. Define the time interval of the time series in deltat
    3. Define the sta/lta detection parameters
        - Default parameters are recommended
    4. Define minimum distance in degrees within which POIs are not subject to cropping
    5. Define minimum amplitude in meters to determine if signal is noise or no arrival is detected or if there is a measurement and crop
    6. Define which filters to apply 


After these definitions, the code will determine which signals to crop and output pkl files to save memory space.

