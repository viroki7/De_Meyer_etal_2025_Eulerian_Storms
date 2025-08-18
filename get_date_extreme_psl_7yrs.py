import glob
import numpy as np
import xarray as xr
from tqdm import tqdm
import os

all_files = glob.glob('/storm/demeyer/TRACKING/EULERIAN/OUTPUTS/*/date_extreme/*.nc4')
subtract_files = glob.glob('/storm/demeyer/TRACKING/EULERIAN/OUTPUTS/*/date_extreme/*_7_years.nc4')
filenames = [file for file in all_files if file not in subtract_files]

for i, filename in enumerate(filenames):

    if os.path.exists(filename.replace('.nc4','_7_years.nc4'))==False:

        ds = xr.open_dataset(filename)

        print('\n',ds.model.values[0])

        new_ds = ds

        for ilat, lat in tqdm(enumerate(ds.lat)):
            for ilon, lon in enumerate(ds.lon):

                pixel = ds.sel(lat=lat, lon=lon)

                lowest_indices = pixel['psl'].argsort()[:7]

                year_non_ext = np.setdiff1d(pixel['year'].values, pixel['year'].values[lowest_indices])

                pixel_nan = xr.where(pixel.year.isin(year_non_ext) == True, np.nan, pixel)

                if ilon == 0:
                    new_ds = pixel_nan
                else:
                    new_ds = xr.concat([new_ds, pixel_nan], dim = 'lon')

            if ilat == 0:
                ds_seven = new_ds
            else:
                ds_seven = xr.concat([ds_seven, new_ds], dim = 'lat')

        ds_seven.to_netcdf(filename.replace('.nc4','_7_years.nc4'))