#!/bin/bash

# Pour avoir la derniere version de cdo
module load utils/cdo

model=MRCC6_noSN
# file=/storm/benoit/data/GEM5/GEM_hourly/uaa_NetCDF/uaa_198001/var_PN_198001.nc4
file=/home/demeyer/DATA/TRACKING/EULERIAN/INPUTS/MRCC6_GEM5_UAA_noSN/PN/1980/01/var_PN_198001.nc4

# model=MIROC6
# file=/tornado/roberge/CMIP6/CMIP6_6hrPt_historical/psl_6hrPlevPt_MIROC6_historical_r1i1p1f1_gn_191001010600-191101010000.nc

outdir=/home/demeyer/TRACKING/EULERIAN/1_PreProcessing/

#cdo -f nc4c -z zip_1 -k grid -b F32 sellonlatbox,219,308,10,60 ${indir}/era5-land_tp_ll_195001_1h.nc4 ${outdir}/era5-land_box_North_America.nc4

cdo griddes ${file}>${outdir}/${model}-global_grid.txt