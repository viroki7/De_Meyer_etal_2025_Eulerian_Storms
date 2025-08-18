import xarray as xr
from sklearn.linear_model import LinearRegression
import glob
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import copy as cp
import warnings
from scipy import stats
warnings.filterwarnings("ignore")

variables       = ['psl','huss']
parameters      = ['total_mae','linear_mae','error','total_aae','min_mae','slope_mae','asymm_mae','min','slope','asymm']
name_parp       = {'total_mae':r'$\bar{p}_{mae}$','linear_mae':r'$\bar{p}_\hat{mae}$','error':r'$\bar{p}_{error}$','total_aae':'$p_\hat{aae}$',
                   'min_mae':r'$p_{peak}^{abs}$','slope_mae':r'$p_{tend}^{abs}$','asymm_mae':r'$p_{asym}^{abs}$','min':r'$p_{peak}$',
                   'slope':r'$p_{tend}$','asymm':r'$p_{asym}$','min_bias':r'$p_{peak}^{bias}$','slope_bias':r'$p_{tend}^{bias}$','asymm_bias':r'$p_{asym}^{bias}$'}
name_parq       = {'total_mae':r'$\bar{q}_{mae}$','linear_mae':r'$\bar{q}_\hat{mae}$','error':r'$\bar{q}_{error}$','total_aae':'$q_\hat{aae}$',
                   'min_mae':r'$q_{peak}^{abs}$','slope_mae':r'$q_{tend}^{abs}$','asymm_mae':r'$q_{asym}^{abs}$','min':r'$q_{peak}$',
                   'slope':r'$q_{tend}$','asymm':r'$q_{asym}$','min_bias':r'$q_{peak}^{bias}$','slope_bias':r'$q_{tend}^{bias}$','asymm_bias':r'$q_{asym}^{bias}$'}
name_parp_bold = {
    'total_mae': r'$\mathbf{p}_{\mathbf{mae}}$',
    'linear_mae': r'$\mathbf{p}_{\mathbf{\hat{mae}}}$',
    'error': r'$\mathbf{p}_{\mathbf{error}}$',
    'total_aae': r'$\mathbf{p}_{\mathbf{\hat{aae}}}$',
    'min_mae': r'$\mathbf{p}_{\mathbf{peak}}^{\mathbf{abs}}$',
    'slope_mae': r'$\mathbf{p}_{\mathbf{tend}}^{\mathbf{abs}}$',
    'asymm_mae': r'$\mathbf{p}_{\mathbf{asym}}^{\mathbf{abs}}$',
    'min': r'$\mathbf{p}_{\mathbf{peak}}$',
    'slope': r'$\mathbf{p}_{\mathbf{tend}}$',
    'asymm': r'$\mathbf{p}_{\mathbf{asym}}$',
    'min_bias': r'$\mathbf{p}_{\mathbf{peak}}^{\mathbf{bias}}$',
    'slope_bias': r'$\mathbf{p}_{\mathbf{tend}}^{\mathbf{bias}}$',
    'asymm_bias': r'$\mathbf{p}_{\mathbf{asym}}^{\mathbf{bias}}$'
}

name_parq_bold = {
    'total_mae': r'$\mathbf{q}_{\mathbf{mae}}$',
    'linear_mae': r'$\mathbf{q}_{\mathbf{\hat{mae}}}$',
    'error': r'$\mathbf{q}_{\mathbf{error}}$',
    'total_aae': r'$\mathbf{q}_{\mathbf{\hat{aae}}}$',
    'min_mae': r'$\mathbf{q}_{\mathbf{peak}}^{\mathbf{abs}}$',
    'slope_mae': r'$\mathbf{q}_{\mathbf{tend}}^{\mathbf{abs}}$',
    'asymm_mae': r'$\mathbf{q}_{\mathbf{asym}}^{\mathbf{abs}}$',
    'min': r'$\mathbf{q}_{\mathbf{peak}}$',
    'slope': r'$\mathbf{q}_{\mathbf{tend}}$',
    'asymm': r'$\mathbf{q}_{\mathbf{asym}}$',
    'min_bias': r'$\mathbf{q}_{\mathbf{peak}}^{\mathbf{bias}}$',
    'slope_bias': r'$\mathbf{q}_{\mathbf{tend}}^{\mathbf{bias}}$',
    'asymm_bias': r'$\mathbf{q}_{\mathbf{asym}}^{\mathbf{bias}}$'
}

param_std       = ['min_std','slope_std','asymm_std']
name_parp_std   = {'min_std':r'$p_{center}$','slope_std':r'$p_{depth}$','asymm_std':r'$p_{asymm}$'}
name_parq_std   = {'min_std':r'$q_{center}$','slope_std':r'$q_{depth}$','asymm_std':r'$q_{asymm}$'}
vars_offset     = {'psl' : 0.01, 'huss' : 1000.} #pour l'unité
path_in         = '/storm/demeyer/TRACKING/EULERIAN/OUTPUTS/'
path_out        = '/storm/demeyer/TRACKING/EULERIAN/OUTPUTS/Figures/'

# Résolution en km prise sur la longitude à l'équateur pour certains modèles et en moyenne sur le globe pour d'autres
# Résolution en degré sur la longitude obtenue depuis la truncation spectrale appliquée si type spectral avec la grille gaussienne équivalente réduite linéairement ou bien le nombre de points de grille pour les modèles à point de grille
# On peut, avec T159 et le nombre de latitude (ou bien l'info que c'est une grille gaussienne car lon=lat), connaitre la grille et donc les résolutions en degré et km à l'équateurs
# Pourquoi pas utiliser la nominal resolution ? https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/asl.952#support-information-section http://goo.gl/v1drZl
# file:///C:/Users/DEMEYER_V.UQAM/Downloads/10.1175_JCLI-D-21-0259.s1.pdf
model_resolution = {
    'ACCESS-CM2': (173.5, 1.53),      # N96 1.875×1.25 https://research.csiro.au/access/about/cm2/
    'ACCESS-ESM1-5': (173.5, 1.53),   # 1.875° longitude by 1.25° https://www.publish.csiro.au/es/fulltext/es19035#:~:text=The%20resolution%20of%20the%20atmospheric,%2DESM1%20and%20ACCESS%2DESM1.
    'AWI-ESM-1-1-LR': (208, 1.88),    # T63L47 native atmosphere T63 gaussian grid; 192 x 96 longitude/latitude https://fesom.de/models/awi-esm/
    'BCC-CSM2-MR': (125, 1.125),      # T106; 320 x 160 longitude/latitude. https://www.cen.uni-hamburg.de/en/icdc/data/cryosphere/cmip6-sea-ice-area/bcc-csm2.html
    'CanESM5': (313, 2.81),           # T63 spectral resolution (approx. 2.8∘) T63L49 native atmosphere, T63 Linear Gaussian Grid; 128 x 64 longitude/latitude  https://www.cen.uni-hamburg.de/en/icdc/data/cryosphere/cmip6-sea-ice-area/canesm5.html
    'CMCC-CM2-SR5': (120, 1.075),     # In CMCC-CM2, CAM5 is used at a horizontal resolution of about 1°, with a regular grid of 0.9° × 1.25° https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018MS001369 1deg; 288 x 192 longitude/latitude https://www.cen.uni-hamburg.de/en/icdc/data/cryosphere/cmip6-sea-ice-area/cmcc-cm2.html
    'CMCC-ESM2': (120, 1.075),        # CMCC-ESM2 has an atmospheric horizontal resolution of about 1°, with a regular grid of 0.9° × 1.25° in latitude and longitude https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002814
    'CNRM-CM6-1-HR': (55, 0.5),       # T359 360 latitude circle https://www.wdc-climate.de/ui/cmip6?input=CMIP6.CMIP.CNRM-CERFACS.CNRM-CM6-1-HR.historical ; 0.5° in the atmosphere https://www.umr-cnrm.fr/cmip6/?article12&lang=en
    'CNRM-CM6-1': (156, 1.4),         # The model horizontal resolution is about 1.4° at the equator. https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019MS001683
    'CNRM-ESM2-1': (156, 1.4),        # T127; Gaussian Reduced with 24572 grid points in total distributed over 128 latitude circles https://www.cen.uni-hamburg.de/en/icdc/data/cryosphere/cmip6-sea-ice-area/cnrm-esm2.html
    'CRCM6-GEM5-UAA-SN': (12, 0.11),  # uses a horizontal grid spacing of 0.11∘ (about 12 km) https://gmd.copernicus.org/articles/17/1497/2024/
    'CRCM6-GEM5-UAA': (12, 0.11),     # same as above
    'CRCM6-GEM5-UBD': (12, 0.11),     # same as above
    'CRCM6-GEM5-UBE': (12, 0.11),     # same as above
    'CRCM6-GEM5-UBF': (12, 0.11),     # same as above
    'EC-Earth3-AerChem': (80, 0.7),   # T255L91 (~80km) https://doi.org/10.5194/gmd-2020-446
    'EC-Earth3-Veg-LR': (126, 1.13),  # T159L62 (~125km) https://doi.org/10.5194/gmd-2020-446
    'EC-Earth3-Veg': (80, 0.7),       # T255L91 (~80km) https://doi.org/10.5194/gmd-2020-446
    'EC-Earth3': (80, 0.7),           # T255L91 (~80km) https://doi.org/10.5194/gmd-2020-446
    'ERA5': (31, 0.25),              
    'GFDL-CM4': (100, 1),             # 1 degree nominal horizontal resolution; 360 x 180 longitude/latitude; https://www.cen.uni-hamburg.de/en/icdc/data/cryosphere/cmip6-sea-ice-area/gfdl.html  roughly 100 km horizontal resolution https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2017MS001208
    'GFDL-ESM4': (100, 1),            # same as above
    'GISS-E2-1-G': (250, 2.24),       # The atmospheric resolution is 2 × 2.5 latitude/longitude https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019MS002025 2.5x2 degree; 144 x 90 longitude/latitude https://www.cen.uni-hamburg.de/en/icdc/data/cryosphere/cmip6-sea-ice-area/giss-e2.html
    'KIOST-ESM': (208, 1.875),        # cubed sphere (C48); 192 x 96 longitude/latitude https://www.cen.uni-hamburg.de/en/icdc/data/cryosphere/cmip6-sea-ice-area/kiost-esm.html
    'MIROC-ES2L': (313, 2.81),        # the horizontal resolution of the atmosphere is set to have T42 spectral truncation, which is approximately 2.8∘ intervals for latitude and longitude. https://gmd.copernicus.org/articles/13/2197/2020/ T42; 128 x 64 longitude/latitude https://www.cen.uni-hamburg.de/en/icdc/data/cryosphere/cmip6-sea-ice-area/miroc-es2l.html
    'MIROC6': (156, 1.4),             # The horizontal resolution is a T85 spectral truncation that is an approximately 1.4∘ grid interval for both latitude and longitude. https://gmd.copernicus.org/articles/12/2727/2019/
    'MPI-ESM-1-2-HAM': (209, 1.875),  # spectral T63; 192 x 96 longitude/latitude https://www.wdc-climate.de/ui/cmip6?input=CMIP6.AerChemMIP.HAMMOZ-Consortium.MPI-ESM-1-2-HAM
    'MPI-ESM1-2-HR': (104, 0.9375),   # spectral T127; 384 x 192 longitude/latitude https://www.cen.uni-hamburg.de/en/icdc/data/cryosphere/cmip6-sea-ice-area/mpi-esm.html
    'MPI-ESM1-2-LR': (209, 1.875),    # spectral T63; 192 x 96 longitude/latitude https://www.cen.uni-hamburg.de/en/icdc/data/cryosphere/cmip6-sea-ice-area/mpi-esm.html
    'MRI-ESM2-0': (125, 1.125),       # TL159; 320 x 160 longitude/latitude https://www.cen.uni-hamburg.de/en/icdc/data/cryosphere/cmip6-sea-ice-area/mri-esm2.html
    'NorESM2-MM': (122, 1.09),        # 1∘ in MM / The “medium-resolution” (M) version has a grid spacing of 1.25∘ × 0.9375∘  https://gmd.copernicus.org/articles/13/6165/2020/ 
    'SAM0-UNICON': (122, 1.09),       # horizontal resolution of 0.95° latitude (lat) × 1.25° longitude (lon) https://journals.ametsoc.org/view/journals/clim/32/10/jcli-d-18-0796.1.xml 1deg; 288 x 192 longitude/latitude https://www.cen.uni-hamburg.de/en/icdc/data/cryosphere/cmip6-sea-ice-area/sam0.html
    'TaiESM1': (122, 1.095)           # 0.9x1.25 degree; 288 x 192 longitude/latitude; https://www.cen.uni-hamburg.de/en/icdc/data/cryosphere/cmip6-sea-ice-area/taiesm1.html
}

grp_mod_res = {'rcm'    : ['CRCM6-GEM5-UAA-SN', 'CRCM6-GEM5-UAA', 'CRCM6-GEM5-UBD', 'CRCM6-GEM5-UBE', 'CRCM6-GEM5-UBF'],
               'high'   : ['CNRM-CM6-1-HR', 'EC-Earth3-AerChem', 'EC-Earth3-Veg', 'EC-Earth3', 'MPI-ESM1-2-HR', 'GFDL-CM4', 'GFDL-ESM4', 'CMCC-CM2-SR5', 'CMCC-ESM2'],
               'medium' : ['NorESM2-MM', 'SAM0-UNICON', 'TaiESM1', 'BCC-CSM2-MR', 'MRI-ESM2-0', 'EC-Earth3-Veg-LR', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'MIROC6'],
               'low'    : ['ACCESS-CM2', 'ACCESS-ESM1-5', 'KIOST-ESM', 'MPI-ESM-1-2-HAM', 'MPI-ESM1-2-LR', 'AWI-ESM-1-1-LR', 'GISS-E2-1-G', 'CanESM5', 'MIROC-ES2L']}

filename = '/storm/demeyer/TRACKING/EULERIAN/OUTPUTS/psl/ACCESS-CM2_NA_1980.nc4'
ds_i = xr.open_mfdataset(filename)
rr = 2
lats = ds_i['lat'].values[rr:-rr]
lons = ds_i['lon'].values[rr:-rr]

tpoints = {'n_atlantic':[317,50],
           'w_canada':[235, 57],
           'e_canada': [287, 57],
           'gulf_mexico': [265, 31],
           'quebec': [360-73.56,45.5],
           'atlantic2': [305, 33],
           'central_us': [256, 38],
           'california': [241, 37],
           'mexico': [260, 24],
           'winnipeg': [265, 47],
           'ontario': [275, 40],
           'alberta': [249, 51],
           'caraibes': [284, 23],
           'pacific': [223, 40],
           'pacific2': [220, 54],
           'manitoba': [261, 58],
           'atlantic3': [290, 37]}

regions = {'n_atlantic': [313, 320, 48, 53],
           'w_canada': [232, 238, 54, 60],
           'e_canada': [286, 293, 55, 60],
           'gulf_mexico': [263, 270, 29, 35],
           'quebec': [285, 292, 44, 49],
           'atlantic2': [302, 308, 31, 36],
           'central_us': [252, 259, 35, 40],
           'california': [238, 245, 35, 40],
           'mexico': [257, 264, 21, 27],
           'winnipeg': [263, 270, 45, 50],
           'ontario': [272, 279, 37, 42],
           'alberta': [244, 251, 47, 53],
           'caraibes': [280, 287, 19, 25],
           'pacific': [220, 227, 38, 43], #peux pas plus bas pour le CRCM6
           'pacific2': [217, 224, 52, 57],
           'manitoba': [258, 265, 55, 60],
           'atlantic3': [285, 292, 33, 39]}


def true_tpoints(tpoints):
    tpoints_true = {}
    for key, (lon, lat) in tpoints.items():
        tpoints_true[key] = [
            lons[np.argmin(np.abs(lons - lon))],
            lats[np.argmin(np.abs(lats - lat))]
        ]
    return tpoints_true

def true_regions(regions):
    regions_true = {}
    for region, coords in regions.items():
        lon_1 = np.min(lons[lons >= coords[0]])
        lon_2 = np.max(lons[lons <= coords[1]])
        lat_1 = np.min(lats[lats >= coords[2]])
        lat_2 = np.max(lats[lats <= coords[3]])
        regions_true[region] = [
            (lon_1 + lons[np.where(lons == lon_1)[0] - 1]) / 2,
            (lon_2 + lons[np.where(lons == lon_2)[0] - 1]) / 2,
            (lat_1 + lats[np.where(lats == lat_1)[0] - 1]) / 2,
            (lat_2 + lats[np.where(lats == lat_2)[0] - 1]) / 2
        ]
    return regions_true

def chiffre_en_lettre(chiffre):
    if 1 <= chiffre <= 26:
        return chr(chiffre + 64)  # 65 est le code Unicode de 'A', donc + 64
    else:
        return None  # Retourne None si le chiffre n'est pas dans l'intervalle [1, 26]
    
def _mode(*args, **kwargs):
    vals = stats.mode(*args, kwargs['axis'], nan_policy='omit')
    if kwargs['mode'] == 'mode':
        return vals[0].squeeze()
    if kwargs['mode'] == 'count':
        return vals[1].squeeze()

def mode(obj, dim, mode):
    # note: apply always moves core dimensions to the end
    # usually axis is simply -1 but scipy's mode function doesn't seem to like that
    # this means that this version will only work for DataArray's (not Datasets)
    assert isinstance(obj, xr.DataArray)
    axis = obj.ndim - 1
    return xr.apply_ufunc(_mode, obj,
                        input_core_dims=[[dim]],
                        kwargs={'axis': axis, 'mode': mode}, dask = 'allowed') 

def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def round_to_2(x):
   return round(x, 1-int(np.floor(np.log10(abs(x)))))

def get_linearfit(tsteps, data, middle):
   model = LinearRegression(fit_intercept=False)
   reg_l = cp.deepcopy(model.fit(tsteps[:middle+1].reshape((-1, 1)), data[:middle+1] - data[middle]))
   reg_r = cp.deepcopy(model.fit(tsteps[middle:].reshape((-1, 1)), data[middle:] - data[middle]))
   cen = data[middle]
   sym = 0.5 * (reg_l.coef_[0] + reg_r.coef_[0])
   slo = 0.5 * (reg_r.coef_[0] - reg_l.coef_[0])
   y = cen + sym * tsteps + slo * np.abs(tsteps)
   return cen, sym, slo, y

def pre_process(background, cyclone, region='False', remove='False', figure='False'):
   
    listf = sorted(glob.glob(path_in + background + '_background/E-parameters_psl-huss_*' + cyclone + '*_top7.npy'), key=lambda x: x.lower())
    # list = listf.remove(path_in+background+'_background/E-parameters_psl-huss_'+cyclone+'_CRCM6-noSN_top7.npy')
    models = []
    for f, ff in enumerate(listf):
       mod = ff.split('/')[-1].split('_')[-2]
       if mod not in models:
          models.append(mod.replace('.npy',''))

    #Remove a model :
    if remove != 'False':
        for mod in remove:
            models.remove(mod)
            listf.remove(path_in+background+'_background/E-parameters_psl-huss_'+cyclone+'_'+mod+'_top7.npy')

    var1d = []
    var2d = []
    for imod, fileout in enumerate(listf):
        storm_serie = np.load(fileout)
        var1d.append(storm_serie[0,:,:])
        var2d.append(storm_serie[1,:,:])
    var1d = np.asarray(var1d)
    var2d = np.asarray(var2d)

    ds = xr.Dataset(data_vars={'huss': (['models','parameters','latitude','longitude'], var2d), 'psl': (['models','parameters','latitude','longitude'], var1d)},
                  coords={'longitude':lons, 'latitude':lats, 'models':models, 'parameters':parameters})

    fig1 = plt.figure(figsize=(4,4), tight_layout = {'pad': 0})
    # plotcrs = ccrs.Robinson(-80)
    plotcrs = ccrs.PlateCarree()
    ax1 = plt.subplot(projection=plotcrs)
    # ax1.set_extent([-160, -30, 90, 15])
    ax1.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())
    ax1.add_feature(cartopy.feature.OCEAN, color='#A5E1FF', zorder=0)
    ax1.add_feature(cartopy.feature.LAKES, color='#A5E1FF', zorder=1, linewidth=0.3, edgecolor='black')
    ax1.add_feature(cartopy.feature.BORDERS, linewidth=0.3, edgecolor='black', zorder=2)
    ax1.add_feature(cartopy.feature.LAND,  color='#CCC2A6', zorder=0, linewidth=0.3, edgecolor='black')
    gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='k', alpha=0.25,
                xlocs=range(-180,180,10), ylocs=range(-90,90,10))
    gl.xlabel_style = {'size': 4, 'color': 'blue'}
    gl.ylabel_style = {'size': 4, 'color': 'red'}
    ax1.coastlines(resolution='50m', linewidth=0.4, color='black')
    ds_plot = xr.Dataset(data_vars = {'flag': (['latitude','longitude'], np.zeros((len(lats),len(lons)))+1.)},
                        coords = {'longitude': lons, 'latitude': lats})

    if region == 'All':
        region_plot = ds_plot
        plt.contourf(region_plot.longitude, region_plot.latitude, region_plot.flag, alpha=0.7, color = 'k', transform=ccrs.PlateCarree())

    else:
        mask = xr.open_dataset('/storm/demeyer/TOOLS/mask_by_countries.nc')
        mask = mask.assign_coords({'longitude':(mask['longitude'] % 360)})
        mask = mask.sortby(mask['longitude'])
        mask = mask.interp(latitude=lats, longitude=lons, method='nearest')

        if region in ['Canada', 'USA']:
            if region == 'Canada': idx = 41
            if region == 'USA': idx = 238
            ds = ds.where(mask.mask == idx, drop=True)
            region_plot = ds_plot.where((mask.mask == idx), drop=True)
            plt.contourf(region_plot.longitude, region_plot.latitude, region_plot.flag, alpha=0.7, color = 'k', zorder=2, transform=ccrs.PlateCarree())

        if region in ['North_America', 'North_America_notropics']:
            if region == 'North_America':
                ds = ds.where((mask.mask == 238) | (mask.mask == 41), drop=True)
                region_plot = ds_plot.where((mask.mask == 238) | (mask.mask == 41), drop=True)
                plt.contourf(region_plot.longitude, region_plot.latitude, region_plot.flag, alpha=0.7, color = 'k', zorder=2, transform=ccrs.PlateCarree())
            if region == 'North_America_notropics':
                lat_max = 65
                lat_min = 30
                ds = ds.where((mask.mask == 238) | (mask.mask == 41), drop=True).sel(latitude=slice(lat_min,lat_max))
                region_plot = ds_plot.where((mask.mask == 238) | (mask.mask == 41), drop=True).sel(latitude=slice(lat_min,lat_max))
                plt.contourf(region_plot.longitude, region_plot.latitude, region_plot.flag, alpha=0.7, color = 'k', zorder=2, transform=ccrs.PlateCarree())

        if type(region) == list:
            ds = ds.sel(longitude=slice(region[0], region[1]), latitude=slice(region[2], region[3]))
            ds = ds.isel(longitude=slice(None, -1), latitude=slice(None, -1))
            region_plot = ds_plot.sel(longitude=slice(region[0], region[1]), latitude=slice(region[2], region[3]))
            plt.contourf(region_plot.longitude, region_plot.latitude, region_plot.flag, alpha=0.7, color = 'k', zorder=2, transform=ccrs.PlateCarree())
            
            # plt.savefig(path_out+str(region[0])+'-'+str(region[1])+'_'+str(region[2])+'-'+str(region[3])+'-region.png', bbox_inches='tight', dpi=300)
    
    if figure=='True':
        plt.show()
    plt.close()

    ds_era = ds.sel(models='ERA5')

    ds_minus_era = ds - ds_era
    ds_minus_era = ds_minus_era.drop_sel(models="ERA5")

    ds_without_era = ds.drop_sel(models="ERA5")
    
    ds_bias = ds_minus_era.mean(dim=['latitude','longitude'])
    ds_err  = ds_without_era.mean(dim=['latitude','longitude'])

    return models, ds_bias, ds_err, ds_era, ds_without_era, ds_minus_era