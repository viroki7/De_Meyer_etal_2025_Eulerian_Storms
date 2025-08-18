from scipy.optimize import leastsq
from sklearn.linear_model import LinearRegression
import os
import xarray as xr
import sys
import glob
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from matplotlib import ticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import copy as cp
import warnings
import dask

warnings.filterwarnings("ignore")

def get_linearfit(tsteps, data, middle):
   model = LinearRegression(fit_intercept=False)
   reg_l = cp.deepcopy(model.fit(tsteps[:middle+1].reshape((-1, 1)), data[:middle+1] - data[middle]))
   reg_r = cp.deepcopy(model.fit(tsteps[middle:].reshape((-1, 1)), data[middle:] - data[middle]))
   cen = data[middle]
   sym = 0.5 * (reg_l.coef_[0] + reg_r.coef_[0])
   slo = 0.5 * (reg_r.coef_[0] - reg_l.coef_[0])
   y = cen + sym * tsteps + slo * np.abs(tsteps)
   return cen, sym, slo, y

"""
This script will calculate a numer of parameters to charaterise the extreme storms identified using the 1_get_synoptic-extreme.py script.
The parameters are based on the double liner fit to the data as explained in the presentation. Two variables are used: the MSLP that was used to identify the extremes and the associated near-surface humidty field.

The characteristics include the minima, the slope and the assymetry.

Authors:
Alejandro Di Luca (di_luca.alejandro@uqam.ca), UQAM-ESCER

Created:
23 june 2022

Last Modified:
23 june 2022

"""

# INPUT DIRECTORY
path_in = '/storm/demeyer/TRACKING/EULERIAN/OUTPUTS/'

# Initialisation
backgrounds = ['no', 'with']
cyclones = ['min','max']
variables = ['psl', 'huss']
vars_offset = {'psl':0.01,'huss':1000.} #pour l'unité
st = 6 #Combien de pas de temps avant t0
tsteps = (np.arange((st * 2) + 1) - st) * 6 #réso tempo de la série temporelle, de 6 en 6 centré sur 0
top = 7
rr = 2
par_names = ['total','approx','error','total_a','min','slope','asymm','min_b','slope_b','asymm_b']

# Read lat/lon
filename1 = glob.glob('/storm/demeyer/TRACKING/EULERIAN/OUTPUTS/psl/ACCESS-CM2_NA_1980.nc4')[0]
ds_i = xr.open_mfdataset(filename1)
lats = ds_i['lat'].values[rr:-rr]
lons = ds_i['lon'].values[rr:-rr]

for cyclone in cyclones:
    print('\n\n###################################################\n\ncyclone: ',cyclone)

    for background in backgrounds:
      print('\n\n#####\n\n\nbackground: ',background)

      # Get model names                                                           
      listf = glob.glob(path_in+background+'_background/E-storms_psl-huss_*'+cyclone+'*.npy')
      models = []
      for ff in listf:
         mod=ff.split('/')[-1].split('_')[-1]
         if mod not in models:
            models.append(mod.replace('.npy',''))
      models.sort()
      models.remove('ERA5')
      models=['CRCM6-UBD','CRCM6-UBE','CRCM6-UBF'] # modif vic
      models.insert(0, 'ERA5')
      
      for imod, mod in enumerate(models):
         print('\nmodel: ',mod)

         file_in = glob.glob(path_in+background+'_background/E-storms_psl-huss_'+cyclone+'*'+mod+'.npy')[0]

         storm_serie = np.load(file_in)[:,:,st:-st,:] #cy ou acy (selon le file) avec mslp et huss, sur les 35 ans, série tempo de +/- 24-6*2j, toutes les lat et lon

         if mod=='ERA5':
            storm_serie_era = cp.deepcopy(storm_serie)

         middle = int((storm_serie.shape[2] - 1) / 2) #milieu de la série tempo
         parameters = cp.deepcopy(storm_serie[:,0,:len(par_names),:]*0)

         file_out = path_in+background+'_background/E-parameters_psl-huss_'+cyclone+'_'+mod+'_top'+str(top)+'.npy'
         
         for ilon,lon in enumerate(lons):
            for ilat,lat in enumerate(lats):
                  
                  for ivar,var in enumerate(variables):

                     if var == 'psl':
                        sorted_storm_serie = np.argsort(storm_serie[ivar,:,middle,ilat,ilon]) #rangement des années dont la valeur du minimum annuel de msl est la plus basse sur ce gridbox
                        sorted_storm_serie_era = np.argsort(storm_serie_era[ivar,:,middle,ilat,ilon])

                     # Calculate for model
                     seven_years_storm_serie = vars_offset[var] * np.nanmean(storm_serie[ivar,sorted_storm_serie[:top],:,ilat,ilon], axis=0) #la moyenne de la série tempo de la msl/huss des 7 années avec la msl la plus basse
                     if np.isnan(seven_years_storm_serie).all() == False:
                        cen, sym, slo, y = get_linearfit(tsteps, seven_years_storm_serie, middle)
                        
                        # Calculate for ERA5
                        seven_years_storm_serie_era = cp.deepcopy(vars_offset[var] * np.nanmean(storm_serie_era[ivar,sorted_storm_serie_era[:top],:,ilat,ilon], axis=0))
                        cen_era, sym_era, slo_era, y_era = get_linearfit(tsteps, seven_years_storm_serie_era, middle)

                        # Calculate errors
                        total   = np.nanmean(np.abs(seven_years_storm_serie - seven_years_storm_serie_era)) #la moyenne de (la série tempo (12*6h) moyenne des 7 années avec la slp la plus basse du modèle - la série tempo moyenne des 7 années avec la slp la plus basse de era5)
                        approx  = np.nanmean(np.abs(y - y_era))                #la moyenne de (la série tempo moyenne des 7 années avec la slp la plus basse du modèle reconstruite par le modèle théorique - la série tempo moyenne des 7 années avec la slp la plus basse de era5 reconstruite par le modèle théorique)
                        min_a   = np.abs(cen - cen_era)                        #le minimum de la série tempo moyenne des 7 années avec la slp la plus basse du modèle - le minimum de la série tempo moyenne des 7 années avec la slp la plus basse de era5
                        slope_a = np.nanmean(np.abs((slo - slo_era) * tsteps)) #la moyenne de (la symétrie par rapport au minimum de la série tempo moyenne des 7 années avec la slp la plus basse du modèle - la symétrie par rapport au minimum de la série tempo moyenne des 7 années avec la slp la plus basse de era5)
                        asymm_a = np.nanmean(np.abs((sym - sym_era) * tsteps)) #la moyenne de (la pente absolue moyenne de chaque côté du minimum de la série tempo moyenne des 7 années avec la slp la plus basse du modèle - la pente absolue moyenne de chaque côté du minimum de la série tempo moyenne des 7 années avec la slp la plus basse de era5)
                        total_a = min_a + slope_a + asymm_a                    #la somme de la différence 3 paramètres de la reconstruction théorique entre le modèle et era5

                        if total != 0:
                           error = np.abs(100. * (approx - total) / (total + approx))     #l'erreur relative entre la différence entre le modèle et era5 et la différence entre le modèle et era5 reconstruite par le modèle théorique
                        else:
                           error = np.nan

                        parameters[ivar,:,ilat,ilon] = [total, approx, error, total_a, min_a, slope_a, asymm_a, cen, slo, sym]

                     else:
                        parameters[ivar,:,ilat,ilon] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                        
         # Save file                                                                                                                
         np.save(file_out, parameters)
         print(file_out,'\n')

         
