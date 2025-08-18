import os
import dask
import sys
import glob
import numpy as np
import datetime as dt
import xarray as xr
import netCDF4 as nc
import h5py
import scipy
from tqdm import tqdm


# from dask.distributed import Client
# client = Client(n_workers=2,threads_per_worker=10,memory_limit='31GB')
# client

# Background
regions  = ['NA']
variable = 'huss' #'psl'
comp     = dict(zlib=True, complevel=5)

# INPUT DIRECTORY
path_dir     = {'psl':'/pampa/roberge/CMIP6/CMIP6_psl_historical/','huss':'/pampa/roberge/CMIP6/CMIP6_huss_historical/'}
path_dir_vdm = {'psl':'/storm/demeyer/TRACKING/EULERIAN/INPUTS/CMIP6_psl_historical/','huss':'/storm/demeyer/TRACKING/EULERIAN/INPUTS/CMIP6_huss_historical/'}
path_dir_vdm_6h = {'psl':'/storm/demeyer/TRACKING/EULERIAN/INPUTS/CMIP6_psl_historical/','huss':'/storm/demeyer/TRACKING/EULERIAN/INPUTS/CMIP6_huss_6h_historical/'}
path_out     = '/storm/demeyer/TRACKING/EULERIAN/OUTPUTS/'+variable+'/'

# DATES
i_date = dt.datetime(1980,1,1,0)
f_date = dt.datetime(2014,12,31,21)
period = [i_date,f_date]
years  = np.arange(i_date.year,f_date.year+1)

# Useful model lists
cftime_list = ['KIOST-ESM','GFDL-ESM4','CMCC-ESM2','SAM0-UNICON','CMCC-CM2-HR4','CMCC-CM2-SR5','FGOALS-f3-L','KACE-1-0-G',
            'NorESM2-MM','TaiESM1','BCC-CSM2-MR','GFDL-CM4','IPSL-CM5A2-INCA','GISS-E2-1-G','FGOALS-g3','CanESM5','IITM-ESM','NorESM2-LM']
black_list  = ['HadGEM3-GC31-MM','HadGEM3-GC31-LL','KACE-1-0-G','FGOALS-g3','IITM-ESM','NorESM2-LM']
models_vdm  = ['NorESM2-MM', 'MIROC-ES2L', 'FGOALS-g3', 'HadGEM3-GC31-MM','IITM-ESM','CNRM-CM6-1-HR','CNRM-CM6-1','HadGEM3-GC31-LL','NorESM2-LM','GFDL-ESM4','GFDL-CM4','TaiESM1']

# OUTPUT GRID
filenames = glob.glob(path_dir['psl']+'psl_6hrPlevPt_MIROC6_historical_r1i1p1f1_gn_191001010600-191101010000.nc')
with xr.open_mfdataset(filenames) as dsbase:
    new_lat = dsbase.sel(lat=slice(10, 80)).lat
    new_lon = dsbase.sel(lon=slice(210, 330)).lon

# Get model names
listf = glob.glob(path_dir[variable]+variable+'_*.nc') + glob.glob(path_dir_vdm[variable]+variable+'_*.nc')
models=[]
for ff in listf:
   mod=ff.split('/')[-1].split('_')[2]
   if mod not in models:
      models.append(mod)

# Start variables loop
for region in regions:
    print('Treating region ',region)
    if region == 'NA':
        array_latlon = [10.,80.,210.,330.]
    if region == 'Arctic':
        array_latlon = [50.,90.,0.,360.]
    if region == 'glob':
        array_latlon = [-90.,90.,0.,360.]

    for imod,mod in enumerate(tqdm(sorted(models))):
        print('\n Processing model ', mod)
        file_out = path_out+mod+'_'+region+'_1980-2014.nc4'
        # if os.path.exists(file_out) == True and mod not in black_list:
        if os.path.exists(file_out) == False and mod not in black_list:

            if mod in ['CNRM-CM6-1-HR','NorESM2-MM']:
                filenames = glob.glob(path_dir_vdm_6h[variable]+variable+'_*'+mod+'_*.nc')
            elif mod in models_vdm:
                filenames = glob.glob(path_dir_vdm[variable]+variable+'_*'+mod+'_*.nc')
            else:
                filenames = glob.glob(path_dir[variable]+variable+'_*'+mod+'_*.nc')
            filenames.sort()

            if mod in ['MPI-ESM-1-2-HAM','GFDL-ESM4','MPI-ESM1-2-LR','CNRM-ESM2-1','CNRM-CM6-1']:
                filenames = filenames[-3:]
            if mod in ['SAM0-UNICON','MIROC6','EC-Earth3-AerChem','EC-Earth3-CC','EC-Earth3-Veg-LR','EC-Earth3-Veg','EC-Earth3','AWI-ESM-1-1-LR','AWI-ESM-1-1-LR','MIROC-ES2L','HadGEM3-GC31-MM','IITM-ESM']:
                filenames = filenames[-35:]
            if mod in ['KACE-1-0-G','BCC-CSM2-MR']:            
                filenames = filenames[-12:]
            if mod in ['CNRM-CM6-1-HR']:    
                filenames = filenames[-17:]    
            if mod in ['CMCC-CM2-HR4','MPI-ESM1-2-HR','CMCC-ESM2','CMCC-ESM2','CMCC-CM2-SR5','FGOALS-f3-L']:            
                filenames = filenames[-7:]              
            if mod in ['GISS-E2-1-G','MPI-ESM-1-2-HAM','MPI-ESM1-2-LR','MRI-ESM2-0','NorESM2-MM','KIOST-ESM','TaiESM1','ACCESS-CM2','ACCESS-ESM1-5','NorESM2-LM']:
                filenames = filenames[-4:]
            if mod in ['CanESM5']:
                filenames = filenames[-5:]    
            if mod in ['NESM3','HadGEM3-GC31-LL','GFDL-CM4']:
                filenames = filenames[-2:]
            if mod in ['IPSL-CM5A2-INCA','IPSL-CM6A-LR']:
                filenames = filenames[-1:]                                  
            if mod in ['KACE-1-0-G']:
                filenames = filenames[-40:]

            if mod == 'CNRM-CM6-1-HR':
                ds = xr.open_mfdataset(filenames, parallel=True, chunks = dict(time=8))
            else:
                ds = xr.open_mfdataset(filenames, parallel=True, chunks = dict(time=13))
           
            huss = ds[variable]


            if mod in cftime_list:
                date_i = i_date.strftime("%Y-%m-%d %H:%M:%S")
                date_f = f_date.strftime("%Y-%m-%d %H:%M:%S")
                iyear  = ds.time.values[0].year
                fyear  = ds.time.values[-1].year
            else:
                date_i = i_date
                date_f = f_date
                iyear  = ds.time.values[0].astype('datetime64[Y]').astype(int)+1970 #retrieve the years of a datetime64 type array
                fyear  = ds.time.values[-1].astype('datetime64[Y]').astype(int)+1970
            print('\n ', iyear, fyear)
            if iyear > 1980: print('\n ###################### ERROR ###################### \n')

            #Uniformiser la grille temporelle ?
            if mod in ['CNRM-CM6-1-HR','NorESM2-MM']:
                huss = huss.sel(time=slice(date_i, date_f)).interp(lat=new_lat, lon=new_lon, method='linear').chunk(dict(time=235))
            else:   
                huss = huss.sel(time=slice(date_i, date_f)).interp(time=ds.sel(time=slice(date_i, date_f)).time[::2], lat=new_lat, lon=new_lon, method='linear').chunk(dict(time=235))
            print('\n ', huss)

            encoding = {var: comp for var in [variable]}
            huss.to_netcdf(path=file_out, encoding=encoding)