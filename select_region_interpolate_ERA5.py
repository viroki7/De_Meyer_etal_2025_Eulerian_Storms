import csv
import os
import sys
import glob
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
# import fluxnet_classes as fc
#import plot_classes as pc
import xarray as xr
import cartopy.crs as ccrs
import netCDF4 as nc
import matplotlib.colors as mcolors
from scipy.signal import savgol_filter
import calendar as cal
import dask

# dask.config.set({"array.slicing.split_large_chunks": True})

"""
This script reads .csv data from the AmeriFlux dataset and process the data to filter those stations that satisfy several criteria:
1. Data belongs to the period of interest given by i_date,f_date
2. Data belongs to the countries of interest given by countries

Authors:
Alejandro Di Luca (di_luca.alejandro@uqam.ca), UQAM-ESCER

Created:
23 june 2022

Last Modified:
23 june 2022

"""
# INPUT DIRECTORY
path_dir='/home/archive/REANALYSES/ERA5/1h/'
# path_out='/pampa/diluca/EulerianStorms/CMIP6-data-1p4-linear/'
path_out='/storm/demeyer/TRACKING_EULERIAN/'
path_out={'msl':'/storm/demeyer/TRACKING_EULERIAN/psl/', 'q2m':'/storm/demeyer/TRACKING_EULERIAN/huss/'}

region='NA'

# DATES
# reference_date=fc.constants.reference_date
i_date=dt.datetime(1980,1,1,0)
f_date=dt.datetime(2014,12,31,21)

# LATITUDE-LONGITUDE LIMITS
array_latlon=[10.,80.,210.,330.]

# VARIABLES
variables=['msl','q2m']
variables=['msl']
var_dic={'msl':'psl','q2m':'huss'}

# OUTPUT GRID
# filenames=glob.glob('/pampa/diluca/EulerianStorms/CMIP6-data/psl/MIROC6*'+region+'*.nc4')
filenames=glob.glob('/storm/demeyer/TRACKING_EULERIAN/psl/MIROC6*'+region+'*.nc4')
ds = xr.open_mfdataset(filenames)
ds.close()
new_lat = ds.lat.values
new_lon = ds.lon.values
new_times=ds.time.values

comp = dict(zlib=True, complevel=5)

# Start variables loop
for ivar,var in enumerate(variables):
      file_out=path_out[var]+'ERA5_'+region+'_'+str(i_date.year)+'-'+str(f_date.year)+'.nc4'
      print(file_out)
      # file_out=path_out+'ERA5_'+region+'_'+str(f_date.year)+'.nc4'
      # ********************************
      # Reading file
      filenames=glob.glob(path_dir+str(var)+'/ll/nc4/*/*/*.nc4')
      filenames.sort()
      ds_i = xr.open_mfdataset(filenames[12:-97], parallel = True, chunks = dict(time=9490, latitude=721, longitude=1440))
      ds_i.close()
      print(ds_i)

      # Selection of region
      mask_lat = ((ds_i.coords["latitude"] > array_latlon[0]) & (ds_i.coords["latitude"] < array_latlon[1]))
      mask_lon = ((ds_i.coords["longitude"] > array_latlon[2]) & (ds_i.coords["longitude"] < array_latlon[3]))
      lat_bd = ds_i.latitude[mask_lat]
      lon_bd = ds_i.longitude[mask_lon]

      regioni = ds_i[var].sel(latitude=lat_bd.values, longitude=lon_bd.values)
      # regioni_period = regioni.sel(time=slice(i_date, f_date))
      # dsi_s = regioni_period.interp(latitude=new_lat,longitude=new_lon,method='linear')
      dsi_s = regioni.interp(latitude=new_lat,longitude=new_lon,method='linear')
      dsi = dsi_s.interp(time=new_times,method='linear')

      # Rename variables and dimensions
      dsi = dsi.rename({'longitude': 'lon','latitude': 'lat', var: var_dic[var]})
      dsi.rename({var: var_dic[var]}) #########################################jsuiscon
      encoding = {var: comp for var in [var]}
      dsi.to_netcdf(path=file_out,format="NETCDF4_CLASSIC",unlimited_dims='time',encoding=encoding,engine='netcdf4')
      print(os.system('ls -lth '+file_out))

