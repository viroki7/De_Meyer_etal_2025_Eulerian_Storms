import glob
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import os

twindow = 4*3 #12 jours avant et 12 après centrée sur le minimum
rr = 2

listf = sorted(glob.glob('/storm/demeyer/TRACKING/EULERIAN/OUTPUTS/with_background/E-parameters_psl-huss_max*_top7.npy'))
models = []
for f, ff in enumerate(listf):
    mod = ff.split('/')[-1].split('_')[-2]
    if mod not in models:
        models.append(mod.replace('.npy',''))

for remove_backgroud in [True, False]:

    print('Remove background : ', remove_backgroud, '\n')

    if remove_backgroud == True:
        path_out = '/storm/demeyer/TRACKING/EULERIAN/OUTPUTS/no_background/date_extreme/'
    else:
        path_out = '/storm/demeyer/TRACKING/EULERIAN/OUTPUTS/with_background/date_extreme/'

    for imod, mod in enumerate(models):

        print('Processing model : ', mod, '\n')

        if os.path.exists(path_out+'Date_with_psl_max_'+mod+'.nc4') == False:

            if mod == 'CRCM6-SN':
                path_in = '/storm/demeyer/TRACKING/EULERIAN/INPUTS/MRCC6_GEM5-UAA-SN_REGRID_MIROC6/'
                filenames = glob.glob(path_in+'/PN/*/*/*_6hr_interp.nc4')
            elif mod == 'CRCM6':
                path_in = '/storm/demeyer/TRACKING/EULERIAN/INPUTS/MRCC6_GEM5-UAA/'
                filenames = glob.glob(path_in+'/PN/*/*/*_6hr_regrid_MIROC6.nc4')
            else:
                path_in = '/storm/demeyer/TRACKING/EULERIAN/OUTPUTS/psl/'
                filenames = glob.glob(path_in+'/'+mod+'_*.nc4')

            ds = xr.open_mfdataset(filenames, parallel=True, chunks = dict(time=30))

            if mod in ['CRCM6-SN', 'CRCM6']: ds = ds.sel(lat=slice(10, 80), lon=slice(210, 330)).isel(time=slice(0, -1)) #Il faut sélectionner les lat/lon ici car pas de preprocessing avec GEM5 et le dernier pas de temps jsp pk mais il faut pour que les dim match
            if mod == 'ERA5': ds = ds.rename({'msl':'psl', 'latitude':'lat', 'longitude':'lon'})

            if remove_backgroud==True:
                yeard = ds['psl'].groupby('time.year').mean('time') #moyenne de l'année par point de grille
                backt = ds['psl'].groupby('time.dayofyear').mean('time') #moyenne journalière par point de grille sur toutes les années
                backt = backt.interp(dayofyear=np.arange(1, backt['dayofyear'].shape[0]+1, 0.25), method='linear', kwargs=dict(fill_value='extrapolate')) #on extrapole chaque 6hr la moyenne journalière. Un moyen d'avoir la moyenne journalière par point de grille sur toutes les années chaque 6hr, mais avec extrapolation de la moyenne journalière.
                
                for iyear, year in enumerate(yeard.year):
                    aa = ds['psl'].sel(time=ds.time.dt.year.isin([year])).shape[0] #nb de donnée dans la ième année
                    time_coords = ds.sel(time=ds.time.dt.year.isin([year])).time.values

                    temp_singleyr = backt.isel(dayofyear=slice(0,aa), lat=slice(rr,-rr), lon=slice(rr,-rr))
                    temp_singleyr = temp_singleyr.assign_coords(time=('dayofyear', time_coords))
                    temp_singleyr = temp_singleyr.swap_dims({'dayofyear': 'time'}).drop('dayofyear')

                    if iyear == 0:
                        temp = temp_singleyr

                    else:
                        temp = xr.concat([temp, temp_singleyr], dim = 'time')

                field = ds['psl'].isel(lat=slice(rr,-rr), lon=slice(rr,-rr)) - temp

            else:
                field = ds['psl'].isel(lat=slice(rr,-rr), lon=slice(rr,-rr))
            
            print('Computing...')
            field = field.compute()
            print('Computing done!')

            for cyclone in ['min','max']:

                if os.path.exists(path_out+'Date_with_psl_'+cyclone+'_'+mod+'.nc4') == False:

                    dict_timeofthemin = []

                    file_out = path_out+'Date_with_psl_'+cyclone+'_'+mod+'.nc4'

                    if cyclone == 'min': print('\nCyclones\n')
                    else: print('\nAnticyclones\n')

                    for iyear, year in tqdm(enumerate(yeard.year.values)):   #en gros, on prend les items temporels un peu avant et après l'année sauf pour 1980 et 2014

                        if year == year.min():       
                            fieldt = xr.concat([field.sel(time=(ds.time.dt.year == year)), field.sel(time=(ds.time.dt.year == year+1)).isel(time=slice(0, twindow))], dim = 'time') #twindow + 1 ??
                            
                        elif year == year.max():
                            fieldt = xr.concat([field.sel(time=(ds.time.dt.year == year-1)).isel(time=slice(-twindow, -1)), field.sel(time=(ds.time.dt.year == year))], dim = 'time')

                        else:
                            fieldt = xr.concat([field.sel(time=(ds.time.dt.year == year-1)).isel(time=slice(-twindow, -1)), field.sel(time=(ds.time.dt.year == year)), field.sel(time=(ds.time.dt.year == year+1)).isel(time=slice(0, twindow))], dim = 'time')
                        
                        if cyclone == 'min':
                            index_min = fieldt.isel(time=slice(twindow,-twindow)).min(dim='time') #le minimum sur l'année en dehors des extrémités de l'année (car la série tempo courerait sur les années suivantes/précédentes) à chaque point lon lat
                            
                        if cyclone == 'max':
                            index_min = fieldt.isel(time=slice(twindow,-twindow)).max(dim='time')

                        for ilat, lat in enumerate(ds['lat'].isel(lat=slice(rr,-rr))):
                            for ilon, lon in enumerate(ds['lon'].isel(lon=slice(rr,-rr))):
                                
                                if np.isnan(fieldt.sel(lon=lon, lat=lat)).all() == False:

                                    tt = fieldt.sel(lon=lon, lat=lat).where(fieldt.sel(lon=lon, lat=lat) == index_min.sel(lon=lon, lat=lat), drop=True).time.values[0] #ce serait mieux de tout faire avec argmin mais la gestion des slices de NaN complique la chose

                                    if isinstance(tt, np.datetime64):
                                        julianday = int(pd.to_datetime(tt).strftime('%j'))
                                        month = int(pd.to_datetime(tt).strftime('%m'))
                                        week = int(pd.to_datetime(tt).strftime('%U'))

                                        if month in [int(12), int(1), int(2)]:
                                            season = int(1)
                                        elif month in [int(3), int(4), int(5)]:
                                            season = int(2)
                                        elif month in [int(6), int(7), int(8)]:
                                            season = int(3)
                                        elif month in [int(9), int(10), int(11)]:
                                            season = int(4)                                    
                                    
                                    else:
                                        julianday = tt.strftime('%j')
                                        month = tt.strftime('%m')
                                        week = tt.strftime('%U')

                                        if month in ['12', '01', '02']:
                                            season = int(1)
                                        elif month in ['03', '04', '05']:
                                            season = int(2)
                                        elif month in ['06', '07', '08']:
                                            season = int(3)
                                        elif month in ['09', '10', '11']:
                                            season = int(4)

                                    dict_timeofthemin.append(
                                        {
                                            'julianday': julianday,
                                            'week': week,
                                            'month': month,
                                            'season': season,
                                            'psl': float(np.round(index_min.sel(lon=lon, lat=lat).values / 100., 2)),
                                            'year': year,
                                            'lon': float(np.round(lon.values, 2)),
                                            'lat': float(np.round(lat.values, 2))
                                        }
                                    )
                                    
                    df = pd.DataFrame(dict_timeofthemin)
                    df = df.set_index(['year','lon','lat'])
                    df.to_csv(path_out+'Date_with_psl_'+cyclone+'_'+mod+'.csv')
                    ds_out = df.to_xarray()
                    ds_out = ds_out.assign_coords(model=('model', [mod]))
                    print('\n\n\n', ds_out, '\n\n\n')
                    ds_out.to_netcdf(path_out+'Date_with_psl_'+cyclone+'_'+mod+'.nc4')