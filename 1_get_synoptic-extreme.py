import csv
import os
import sys
import glob
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import netCDF4 as nc
import matplotlib.colors as mcolors
from scipy.signal import savgol_filter
import calendar as cal

"""
This script identifies extreme "storms" by identifying the minimum 6-hourly mean sea level pressure (MSLP) at each grid point and each year.
There are a few option to make calculations using the raw MSLP (with background) or by remvoing the background MSLP and finding the minima of the anomalies. This is controlled with the remove_backgroud variable.

The script loops through the ERA5 reanalysis and CMIP6 models that were already pre-processed.

I have also added the ability to find extreme anticyclones. This is controlled by the "cyclones" variable and will search for cyclones when cyclone='min' and for anticyclones when cyclone='max'.

Authors:
Alejandro Di Luca (di_luca.alejandro@uqam.ca), UQAM-ESCER

Created:
23 june 2022

Last Modified:
23 june 2022

"""

# Background
variables=['psl','huss']
var1=variables[0]
var2=variables[1]

# Select region
region='NA'

# INPUT DIRECTORY
path_in='/pampa/diluca/EulerianStorms/CMIP6-data-1p4-linear/'+var1+'/'

# Get model names as the intersection between models available for vars 1 and 2                                                                
listf=glob.glob(path_in.replace(var1,var1)+'*'+region+'*.nc4')
models1=[]
for ff in listf:
   mod=ff.split('/')[-1].split('_')[0]
   if mod not in models1:
      models1.append(mod)

listf=glob.glob(path_in.replace(var1,var2)+'*'+region+'*.nc4')
models2=[]
for ff in listf:
   mod=ff.split('/')[-1].split('_')[0]
   if mod not in models2:
      models2.append(mod)
models=list(set(models1) & set(models2))

# Cyclone type: cyclone (min) and anticyclone (max)
cyclones=['min','max']

# Start variables loop
ww=4*3
rr=2
for remove_backgroud in [False,True]:
   path_out='/pampa/diluca/EulerianStorms/with-background/'
   if remove_backgroud==True:
      path_out='/pampa/diluca/EulerianStorms/no-background/'
   for imod,mod in enumerate(models[:]):
      print('Processing model ',mod)
      file_out=path_out+'E-storms_'+var1+'-'+var2+'_max_'+mod+'.npy'
      if os.path.exists(file_out)==False:
         # ********************************                                                                                           
         # Reading file                                                                                                             
         filenames=glob.glob(path_in+'/'+mod+'_*.nc4')[0]
         print(filenames)
         ds = xr.open_mfdataset(filenames)
         ds.close()
         yeard=ds['psl'].groupby('time.year').mean('time')
         if remove_backgroud==True:
            backt=ds['psl'].groupby('time.dayofyear').mean('time')
            backt=backt.interp(dayofyear=np.arange(1,backt['dayofyear'].shape[0]+1,0.25),method='linear', kwargs=dict(fill_value='extrapolate')).values
            for iyear,year in enumerate(yeard.year.values):
               aa=ds['psl'].sel(time=ds.time.dt.year.isin([year])).shape[0]
               if iyear==0:
                  temp=backt[np.arange(aa),rr:-rr,rr:-rr]
               else:
                  temp=np.concatenate((temp,backt[np.arange(aa),rr:-rr,rr:-rr]),axis=0)
            field=ds['psl'].values[:,rr:-rr,rr:-rr]-temp
         else:
            field=ds['psl'].values[:,rr:-rr,rr:-rr]

         ds2 = xr.open_mfdataset(filenames.replace(var1,var2))
         ds2.close()
         field2=ds2[var2].values[:,rr:-rr,rr:-rr]
      
         years=yeard.year.values
         mslp=np.zeros((len(variables),len(cyclones),years.shape[0],1+int(2*ww),field.shape[1],field.shape[2]))
         for iyear,year in enumerate(years):   
            aa0=np.where(ds.time.dt.year.isin([year]).values)[0].min()-ww-1
            aa1=np.where(ds.time.dt.year.isin([year]).values)[0].max()+ww+1
            print(year,aa0,aa1)
            fieldt=field[aa0:aa1,:,:]
            field2t=field2[aa0:aa1,:,:]
            if year==years.min():
               fieldt=field[:aa1,:,:]
               field2t=field2[:aa1,:,:]
            if year==years.max():
               fieldt=field[aa0:,:,:]
               field2t=field2[aa0:,:,:]

            # Calculate minimum
            for icyc,cyclone in enumerate(cyclones):
               if cyclone=='min':
                  index_min=np.min(fieldt[ww:-ww,:,:],axis=0)
               else:
                  index_min=np.max(fieldt[ww:-ww,:,:],axis=0)

               for ilat,lat in enumerate(ds['lat'].values[rr:-rr]):
                  for ilon,lon in enumerate(ds['lon'].values[rr:-rr]):
                     tt=np.where(fieldt[:,ilat,ilon]==index_min[ilat,ilon])[0]
                     if len(tt)>1:
                        tt=tt[0]
                     indices_t=np.arange(tt-ww,tt+ww+1)
                     #time_mslp=savgol_filter(fieldt[indices_t[:],ilat,ilon], 11, 2)
                     #time2=savgol_filter(field2t[indices_t[:],ilat,ilon], 11, 2)
                     time_mslp=fieldt[indices_t[:],ilat,ilon]
                     time2=field2t[indices_t[:],ilat,ilon]
                     mslp[0,icyc,iyear,:,ilat,ilon]=time_mslp
                     mslp[1,icyc,iyear,:,ilat,ilon]=time2
                     
         # Save file
         np.save(file_out.replace('max','min'),mslp[:,0,:])
         np.save(file_out,mslp[:,1,:])

