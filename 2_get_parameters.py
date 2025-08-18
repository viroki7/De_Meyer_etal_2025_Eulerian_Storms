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

warnings.filterwarnings("ignore")

def get_linearfit(tsteps,data,middle):
   model = LinearRegression(fit_intercept=False)
   reg_l = cp.deepcopy(model.fit(tsteps[:middle+1].reshape((-1, 1)), data[:middle+1]-data[middle]))
   reg_r = cp.deepcopy(model.fit(tsteps[middle:].reshape((-1, 1)), data[middle:]-data[middle]))
   cen=data[middle]
   sym=0.5*(reg_l.coef_[0]+reg_r.coef_[0])
   slo=0.5*(reg_r.coef_[0]-reg_l.coef_[0])
   y=cen+sym*tsteps+slo*np.abs(tsteps)
   return cen,sym,slo,y

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
# path_in='/pampa/diluca/EulerianStorms/'
path_in='/storm/demeyer/TRACKING/EULERIAN/OUTPUTS/'
path_out='/storm/demeyer/TRACKING/EULERIAN/OUTPUTS/Figures/'

# VARIABLES
backgrounds=['no','with']
variables=['psl','huss']
var_names={'psl':'$p_{top7}$','huss':'$q_{top7}$'}
var1=variables[0]
var2=variables[1]
region='NA'
cyclones=['min','max']
vars_col={'psl':'r','huss':'b'}
vars_offset={'psl':0.01,'huss':1000.} #pour l'unité
years=np.arange(1980,2015)
st=6
tsteps=(np.arange(25-2*st)-st)*6 #réso tempo de la série temporelle, de 6 en 6 centré sur 0
top=7
rr=2
font = {'family':'normal','size': 16}
plt.rc('font', **font)

# Read lat/lon
filename1=glob.glob('/storm/demeyer/TRACKING/EULERIAN/OUTPUTS/psl/ACCESS-CM2_NA_1980.nc4')[0]
ds_i = xr.open_mfdataset(filename1)
ds_i.close()
lats = ds_i['lat'].values[rr:-rr]
lons = ds_i['lon'].values[rr:-rr]
tpoints={'montreal':[360.-73.5,45.5],'miami':[360.-80.2,25.5],'edmonton':[360.-113.5,53.5],'north-atlantic':[360.-43.,55.]}
points=tpoints.keys() #useless?
par_names=['total','approx','error','total_a','min','slope','asymm','min_b','slope_b','asymm_b']

for cyclone in cyclones:
   print('\n'),print('cyclone: ',cyclone)
   for background in backgrounds:
      # background='with' #################################################
      print('background: ',background)
      
      # Get model names                                                           
      listf=glob.glob(path_in+background+'_background/E-storms_psl-huss_*'+cyclone+'*.npy')
      models=[]
      for ff in listf:
         mod=ff.split('/')[-1].split('_')[-1]
         if mod not in models:
            models.append(mod.replace('.npy',''))
      models.sort()
      models.remove('ERA5')
      models.insert(0,'ERA5')
      print(models)
      
      
      for imod, mod in enumerate(models[:]):
         print('\n'),print('model: ',mod)
         file_in=glob.glob(path_in+background+'_background/E-storms_psl-huss_'+cyclone+'*'+mod+'*.npy')[0]
         aa=np.load(file_in)[:,:,st:-st,:] #cy ou acy (selon le file) avec mslp et huss, sur les 35 ans, série tempo de +/- 6j, toutes les lat et lon
         print(np.count_nonzero(np.isnan(aa)),'NaN')
         if mod=='ERA5':
            aa_era=cp.deepcopy(aa)
         middle=int((aa.shape[2]-1)/2) #milieu de la série tempo
         parameters=cp.deepcopy(aa[:,0,:len(par_names),:]*0)
         file_out=path_in+background+'_background/E-parameters_'+var1+'-'+var2+'_'+cyclone+'_'+mod+'_top'+str(top)+'.npy'
         for ivar,var in enumerate(variables):
               print('var: ',var)
               for ilon,lon in enumerate(lons):
                  for ilat,lat in enumerate(lats):
                     if var=='psl':
                        bb=np.argsort(aa[ivar,:,middle,ilat,ilon]) #rangement des années dont la valeur du minimum annuel de msl est la plus basse sur ce gridbox
                        bb_era=np.argsort(aa_era[ivar,:,middle,ilat,ilon])

                     # Calculate for model
                     data=vars_offset[var]*np.nanmean(aa[ivar,bb[:top],:,ilat,ilon],axis=0) #la moyenne de la série tempo de la msl/huss des 7 années avec la msl la plus basse
                     if np.isnan(data).all() == False:
                        cen,sym,slo,y=get_linearfit(tsteps,data,middle)
                     
                     # Calculate for ERA5
                     data_era=cp.deepcopy(vars_offset[var]*np.nanmean(aa_era[ivar,bb_era[:top],:,ilat,ilon],axis=0))
                     cen_era,sym_era,slo_era,y_era=get_linearfit(tsteps,data_era,middle)

                     # Calculate errors
                     total=np.nanmean(np.abs(data-data_era))
                     approx=np.nanmean(np.abs(y-y_era))
                     min_a=np.nanmean(np.abs(cen-cen_era))
                     slope_a=np.nanmean(np.abs((slo-slo_era)*tsteps))
                     asymm_a=np.nanmean(np.abs((sym-sym_era)*tsteps))
                     total_a=min_a+slope_a+asymm_a
                     if total != 0:
                        error=100.*(total-approx)/total
                     else:
                        error=np.nan
                     parameters[ivar,:,ilat,ilon]=[total,approx,error,total_a,min_a,slope_a,asymm_a,cen,slo,sym]
                     for point in points:
                        dist=abs(lon-tpoints[point][0])+abs(lat-tpoints[point][1])
                        if dist<1.0: #si on est sur le bon gridbox
                           fig_name=path_out+mod+'_'+point+'_'+var+'_'+cyclone+'_'+background+'.png'
                           fig, ax = plt.subplots()
                           if var=='psl':
                              ax.set_ylabel(var+' (hPa)', color='k')
                              #if point in ['montreal','edmonton']:
                              #   plt.ylim([-50,25])
                              zmin=np.round(10*data_era.min())/10.-10
                              zmax=np.round(10*data_era.max())/10.+10
                           else:
                              ax.set_ylabel(var+' (g/kg)', color='k')
                              #if point in ['montreal','edmonton']:
                              zmin=np.round(10*data_era.min())/10.-4
                              if zmin<0:
                                 zmin=0
                              zmax=np.round(10*data_era.max())/10.+4
                           
                           plt.ylim([zmin,zmax])
                           ax.text(0.02,0.97,point,bbox = dict(facecolor = 'white', alpha = 0.9),horizontalalignment='left', verticalalignment='top',transform = ax.transAxes)
                           ax.plot(tsteps,data_era,c='k',linewidth=1.5,marker='o')
                           ax.plot(tsteps,data,c=vars_col[var],linewidth=2.,marker='o',label=var_names[var])
                           for ll in range(aa.shape[1]): #pour chaque année
                              ax.plot(tsteps,vars_offset[var]*aa[ivar,ll,:,ilat,ilon],c='grey',linewidth=.2)
                           ax.spines['top'].set_visible(False)
                           ax.spines['right'].set_visible(False)
                           label2=r'$\hat{y}(t) = b+a^{\prime}t+a^{\prime\prime}|t|$'
                           plt.plot(tsteps, y,color='orange',linewidth=2.,linestyle='dotted',label=label2)
                           plt.legend()
                           plt.xlabel('Time from minimum (h)')
                           plt.savefig(fig_name,bbox_inches='tight',dpi=300)
                           #plt.show()
                           plt.close()
                           # print(fig_name)
                           
         # Save file                                                                                                                
         np.save(file_out,parameters)
         # print(file_out)

         
