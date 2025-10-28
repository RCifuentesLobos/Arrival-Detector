
# Arrival-Detector

Codes for detecting the wave arrival for .nc files outputs from HySEA

## Features

Arrival_detector contains codes for estimating the index of the wave arrival in an elevation time series and cropping the leading zero-elevation array, given by a HySEA simulation.  
It does this by combinating two approaches. An STA/LTA algorithm based on automatic phase picking algorithms used in seismology and a coarse travel time estimation using the long wave phase velocity and source-to-destination distance.  
It also contains codes for checking the detection and correcting it in case that both approaches fail.  

It loops through all .nc files in a given directory. If the main directory given as input contains more directories, it loops through all subdirectories looking for .nc files to be processed.  

It outputs compressed pickle files with only non-zero elevation arrays.

## Repository layout

```text
Arrival-Detector
│
├── README.md                                                ← Documentation 
│
├── arrival_detector/                                        ← Main directory
│  │
│  ├── __init__.py
│  ├── arrival_detector_stalta.py                            ← Main code
│  ├── arrival_detector_stalta_mercalli_failsafe.py          ← Non-up-to-date version with command-line parsers
│  ├── HySEAio.py                                            ← i/o functions for HySEA files
│  ├── greatcircle.py                                        ← Geodetic functions
│  ├── POIs.py                                               ← Define POI class
│  ├── detectutils.py                                        ← Detection functions
│  │
│  └── old/                                                  ← Old, legacy codes
│       ├── detecutils.py 
│       ├── arrivals.py
│       ├── arrivals_tt_first.py
│       └── arrival_detector.py      
│
└── Tests/                                                   ← Test directory with non-up-to-date codes
    ├── arrival_detector_stalta_mercalli.py
    ├── arrival_detector_stalta_mercalli_test.py
    └── arrival_detector_stalta_mercalli_test_failsafe.py
```

### Main files description

1. arrival_detector_stalta.py
    - Main functions for looping through simulation outputs detecting arrivals and cropping elevation time series.
        - arrival_detector_stalta.py performs an sta/lta wave picking and complements incorrectly detected waves with travel-time estimations detections.
2. HySEAio.py
    - Contains functions for reading and writing .nc files and pickle files
3. greatcircle.py
    - Contains functions for computing distances and calculating travel times.
4. POIs.py
    - Defines the POI class. Contains methods for visualizing the POIs contained in the input .nc file.
5. detectutils.py  
    - contains functions for the detection, checking and correction of wave arrivals, as well as cropping the time series

### How to use the main function arrival_detector_stalta.py

1. Define the directory with the outputs in dir_out
2. Define the time interval of the time series in deltat
3. Define the sta/lta detection parameters
    - Default parameters are recommended

