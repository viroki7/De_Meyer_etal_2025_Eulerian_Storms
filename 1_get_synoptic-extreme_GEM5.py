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
import dask
import warnings
import argparse

warnings.filterwarnings("ignore")
dask.config.set({"array.slicing.split_large_chunks": True})

# try:
#     client
# except:
#     from dask.distributed import Client
#     client = Client(n_workers=1,threads_per_worker=5,memory_limit='62GB')
# print(client)

parser = argparse.ArgumentParser(description='Regrid netCDF files temporally from 3 hours to 6 hours.')
parser.add_argument('--simu', type=str, required=True, help='Simulation name ("UBD", "UBE" or "UBF")')

args = parser.parse_args()
simu = args.simu

path_in = '/storm/demeyer/TRACKING/EULERIAN/INPUTS/MRCC6_GEM5-'+simu+'/'


variable = {'RPN': ('PN','HU'),
            'NC' : ('psl','huss')}
# variable = {'RPN':'PN','NC':'psl'}
path_in = '/storm/demeyer/TRACKING/EULERIAN/INPUTS/MRCC6_GEM5-'+simu+'/'
cyclones=['min','max']
twindow=4*3 #12 jours avant et 12 après centrée sur le minimum
rr=2

for remove_backgroud in [False,True]:
   print('rmbckgd')
# for remove_backgroud in [False]:
   path_out='/storm/demeyer/TRACKING/EULERIAN/OUTPUTS/with_background/'
   if remove_backgroud==True:
      path_out='/storm/demeyer/TRACKING/EULERIAN/OUTPUTS/no_background/'
   file_out=path_out+'E-storms_'+variable['NC'][0]+'-'+variable['NC'][1]+'_max_CRCM6-'+simu+'.npy'
   if os.path.exists(file_out)==False:
      # filenames = glob.glob(path_in+'/'+variable['RPN'][0]+'/*/*/*_6hr_interp.nc4') #################
      filenames = glob.glob(path_in+'/'+variable['RPN'][0]+'/*/*/*_6hr_regrid_MIROC6.nc4')
      filenames.sort()
      ds = xr.open_mfdataset(filenames, parallel=True, chunks = dict(time=30))
      ds = ds.sel(lat=slice(10, 80), lon=slice(210, 330)).isel(time=slice(0, -1)) #Il faut sélectionner les lat/lon ici car pas de preprocessing avec GEM5 et le dernier pas de temps jsp pk mais il faut pour que les dim match
      yeard=ds[variable['NC'][0]].groupby('time.year').mean('time')#moyenne de l'année par point de grille
      if remove_backgroud==True:
         backt=ds[variable['NC'][0]].groupby('time.dayofyear').mean('time')#moyenne journalière par point de grille sur toutes les années
         backt=backt.interp(dayofyear=np.arange(1,backt['dayofyear'].shape[0]+1,0.25),method='linear', kwargs=dict(fill_value='extrapolate')).values #on extrapole chaque 6hr la moyenne journalière. Un moyen d'avoir la moyenne journalière par point de grille sur toutes les années chaque 6hr, mais avec extrapolation de la moyenne journalière.
         # backt=backt.interp(dayofyear=np.arange(1,backt['dayofyear'].shape[0]+1,1/24),method='linear', kwargs=dict(fill_value='extrapolate')).chunk(dict(dayofyear=30)) #on extrapole chaque 1hr la moyenne journalière. Un moyen d'avoir la moyenne journalière par point de grille sur toutes les années chaque 1hr, mais avec extrapolation de la moyenne journalière.
         for iyear,year in enumerate(yeard.year.values):
            aa=ds[variable['NC'][0]].sel(time=ds.time.dt.year.isin([year])).shape[0] #nb de donnée dans la ième année
            if iyear==0:
               temp=backt[np.arange(aa),rr:-rr,rr:-rr] #toutes les données sauf les frontières géo
            else:
               temp=np.concatenate((temp,backt[np.arange(aa),rr:-rr,rr:-rr]),axis=0) #pour faire une array pour toutes les années
               # temp=xr.concat([temp,backt[np.arange(aa),rr:-rr,rr:-rr]], dim = 'dayofyear') #pour faire une array pour toutes les années
         field=ds[variable['NC'][0]].values[:,rr:-rr,rr:-rr]-temp #
      else:
         field=ds[variable['NC'][0]].values[:,rr:-rr,rr:-rr]

      
      # ds2 = xr.open_mfdataset([items.replace(var1,var2) for items in filenames])
      # filenames = glob.glob(path_in+'/'+variable['RPN'][1]+'/*/*/*_6hr_interp.nc4') #################
      filenames = glob.glob(path_in+'/'+variable['RPN'][1]+'/*/*/*_6hr_regrid_MIROC6.nc4')
      filenames.sort()
      ds2 = xr.open_mfdataset(filenames, parallel=True, chunks = dict(time=30))
      ds2 = ds2.sel(lat=slice(10, 80), lon=slice(210, 330)).isel(time=slice(0, -1))
      field2=ds2[variable['NC'][1]].values[:,rr:-rr,rr:-rr]

      
      years=yeard.year.values
      mslp=np.zeros((len(variable['NC']),len(cyclones),years.shape[0],1+int(2*twindow),field.shape[1],field.shape[2]))+np.nan

      for iyear,year in enumerate(years):   #en gros, on prend les items temporels un peu avant et après l'année sauf pour 1980 et 2014
         print(year)
         aa0=np.where(ds.time.dt.year.isin([year]).values)[0].min()-twindow-1 #coordonnée du premier item de l'année year - la window
         aa1=np.where(ds.time.dt.year.isin([year]).values)[0].max()+twindow+1
         # aa0=np.where(ds.time.dt.year.isin([year]))[0].min()-twindow-1 #coordonnée du premier item de l'année year - la window
         # aa1=np.where(ds.time.dt.year.isin([year]))[0].max()+twindow+1
         # print(year,aa0,aa1)
         if year==years.min():
            fieldt=field[:aa1,:,:]
            field2t=field2[:aa1,:,:]
         elif year==years.max():
            fieldt=field[aa0:,:,:]
            field2t=field2[aa0:,:,:]
         else:
            fieldt=field[aa0:aa1,:,:]
            field2t=field2[aa0:aa1,:,:]

         # Calculate minimum
         for icyc,cyclone in enumerate(cyclones):
            if cyclone=='min':
               index_min=np.min(fieldt[twindow:-twindow,:,:],axis=0) #le minimum sur l'année en dehors des extrémités de l'année (car la série tempo courerait sur les années suivantes/précédentes) à chaque point lon lat
            else:
               index_min=np.max(fieldt[twindow:-twindow,:,:],axis=0)

            for ilat,lat in enumerate(ds['lat'].values[rr:-rr]):
               for ilon,lon in enumerate(ds['lon'].values[rr:-rr]):
                  tt=np.where(fieldt[:,ilat,ilon]==index_min[ilat,ilon])[0] #coordonnée de l'item temporel qui a le miminum sur l'année et à la coord ilon ilat
                  if np.isnan(fieldt[:,ilat,ilon]).all() == False:
                     if len(tt)>1:
                        tt=tt[0]
                     indices_t=np.arange(tt-twindow,tt+twindow+1) #série tempo centrée sur le mini
                     ##time_mslp=savgol_filter(fieldt[indices_t[:],ilat,ilon], 11, 2)
                     ##time2=savgol_filter(field2t[indices_t[:],ilat,ilon], 11, 2)
                     time_mslp=fieldt[indices_t[:],ilat,ilon] #série tempo de la mslp centrée sur le mini
                     time2=field2t[indices_t[:],ilat,ilon] #série tempo de la huss centrée sur le mini
                     mslp[0,icyc,iyear,:,ilat,ilon]=time_mslp
                     mslp[1,icyc,iyear,:,ilat,ilon]=time2
                     #mslp ou huss correspond au mini ou maxi de mslp sur 24 jours centré sur le mini/maxi pour chaque longi lati de chaque année avec le huss correspondant
      # Save file
      print(np.count_nonzero(np.isnan(mslp)),'NaN')
      np.save(file_out.replace('max','min'), mslp[:,0,:])
      np.save(file_out, mslp[:,1,:])

# os.remove('/storm/demeyer/TRACKING/EULERIAN/OUTPUTS/with_background/E-storms_psl-huss_max_GEM5_6hr_interp.npy')
# os.remove('/storm/demeyer/TRACKING/EULERIAN/OUTPUTS/no_background/E-storms_psl-huss_max_GEM5_6hr_interp.npy')