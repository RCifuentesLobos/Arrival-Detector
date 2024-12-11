import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as GHHSFeature
import cartopy.feature as cfeature
import numpy as np
from HySEAio import read_outputs, get_fault_attributes, detect_invalid_values
from greatcircle import haversine, estimate_travel_time
import os
import glob


class POIs(object):
    def __init__(self, filename):
        # filename
        self.filename = filename
        # read the file and set variables
        time, lon, lat, eta, \
            mask_time, mask_lon, mask_lat, mask_eta, \
            fault_info = read_outputs(filename)
        self.time = time
        self.lon = lon
        self.lat = lat
        self.eta = eta
        self.mask_time = mask_time
        self.mask_lon = mask_lon
        self.mask_lat = mask_lat
        self.mask_eta = mask_eta
        self.fault_info = fault_info
        # get fault attributes
        faults = get_fault_attributes(fault_info)
        self.faults = faults

    # Plot a map showing the location of the points of interest and the fault(s)
    def plot_locations(self, plotdistance: bool = False):
        """
        Creates a map of the location of the POIs
        """
        # initialize map
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
                               figsize=(6, 5),
                               dpi=200)
        # colors
        ax.stock_img()
        ax.coastlines(zorder=1)
        maxlat = 80
        minlat = -60
        maxlon = 340
        minlon = 60
        ax.set_extent([minlon, maxlon, minlat, maxlat], crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, alpha=0.5,
                          linestyle='--', color='gray')
        gl.top_labels = False
        gl.right_labels = False

        # plot subfaults
        for i in self.faults:
            f = ax.scatter(i['lon_barycenter'], i['lat_barycenter'],
                           s=20, marker='s', edgecolor='black', facecolor='red', transform=ccrs.PlateCarree(), zorder=10)

        # scale colors of POIs according to the distance to fault
        if plotdistance:
            # 1) compute distance (for now, with only the first fault)
            lonfault = self.faults[0]['lon_barycenter']
            latfault = self.faults[0]['lat_barycenter']
            distance = np.zeros_like(self.lon)
            for i, (lo, la) in enumerate(zip(self.lon, self.lat)):
                distance[i] = haversine(lo, la, lonfault, latfault)
            # plot POIs
            inv_idx, _ = detect_invalid_values(self.mask_eta, self.eta)
            valid_poi = ax.scatter(np.delete(self.lon, inv_idx), np.delete(self.lat, inv_idx),
                                   c=np.delete(distance, inv_idx),
                                   s=10, marker='o', edgecolor='black', linewidth=0.2,
                                   transform=ccrs.PlateCarree(), zorder=3)
            plt.rcParams['hatch.linewidth'] = 0.1
            invalid_poi = ax.scatter(self.lon[inv_idx], self.lat[inv_idx],
                                     s=10, marker='o', edgecolor='black', linewidth=0.2, facecolor='yellow',
                                     hatch='x',
                                     transform=ccrs.PlateCarree(), zorder=3)
        elif not plotdistance:
            # plot POIs
            inv_idx, _ = detect_invalid_values(self.mask_eta, self.eta)
            valid_poi = ax.scatter(np.delete(self.lon, inv_idx), np.delete(self.lat, inv_idx),
                                   s=10, marker='o', edgecolor='black', linewidth=0.2, facecolor='blue',
                                   transform=ccrs.PlateCarree(), zorder=3)
            plt.rcParams['hatch.linewidth'] = 0.1
            invalid_poi = ax.scatter(self.lon[inv_idx], self.lat[inv_idx],
                                     s=10, marker='o', edgecolor='black', linewidth=0.2, facecolor='yellow',
                                     hatch='x',
                                     transform=ccrs.PlateCarree(), zorder=3)
        # labes and titles
        labels = ['Fault',
                  'Valid POI',
                  'Invalid POI']
        plt.legend([f,
                    valid_poi,
                    invalid_poi], labels,
                   loc='lower left', fancybox=True)
        plt.title('POIs locations', fontsize=11)
        plt.tight_layout()
        plt.show()

    # plot the elevation time series of a POI
    def plot_elevation(self, npoi: int):
        """
        plots the npoi-th elevation time series from the output
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.time, self.eta[:, npoi],
                color='k', label=f"POI: {npoi}\n"+r"$\eta$ (m)")
        ax.legend(loc='best')
        plt.xlabel('Time (s)')
        plt.ylabel('Elevation (m)')
        plt.show()
