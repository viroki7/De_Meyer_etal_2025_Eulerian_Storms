#!/bin/bash

# Pour avoir la dernière versions de cdo
module load utils/cdo

simu=$1
var=$2

path=/home/demeyer/DATA/TRACKING/EULERIAN/INPUTS/MRCC6_GEM5-${simu}

startyear=1980
endyear=2014

months="01 02 03 04 05 06 07 08 09 10 11 12"

grille_orig=/home/demeyer/TRACKING/EULERIAN/1_PreProcessing/MRCC6-global_grid.txt
grille_MIROC=/home/demeyer/TRACKING/EULERIAN/1_PreProcessing/MIROC6-global_grid.txt

# var=HU
# var=PN

#mkdir -p ${destination}/${var}

  # Loop over all the years
  year=${startyear}
  while [ ${year} -le ${endyear} ] ; do

       #Display current year
      echo "Traitement de  "
      echo ${year}
      echo " " 

    # Loop over all the months
    for month in ${months} ; do

      #Display current month
      echo "Traitement de  "
      echo ${year}_${month}
      echo " "

      # arch_orig=${path}/uaa_${year}${month}
      arch_orig=${path}/${var}/${year}/${month}

      # Destination
      # arch_new=/storm/demeyer/TRACKING/EULERIAN/INPUTS/MRCC6_GEM5_UAA_REGRID_MIROC6/${var}/${year}/${month}
      arch_new=/home/demeyer/DATA/TRACKING/EULERIAN/INPUTS/MRCC6_GEM5-${simu}/${var}/${year}/${month} #same as arch_orig here
      mkdir -p ${arch_new}
      
      #cdo -L -f nc4c -z zip_1 -k grid -b F32  remapycon,${grille_era5} ${arch_orig}/${model}_${exp_id}_${member_id}_global_${var}_ll_${year}_${month}_r.nc4  ${arch_new}/${model}_${exp_id}_${member_id}_global_${var}_ll_${year}_${month}_interp_ERA5.nc4
      
      cdo -L -f nc4c -z zip_1 -k grid -b F32  remapbil,${grille_MIROC} -setgrid,${grille_orig} ${arch_orig}/var_${var}_${year}${month}.nc4  ${arch_new}/var_${var}_${year}${month}_regrid_MIROC6.nc4
      
    done


    year=$(( $year + 1 ))
    echo ${year}
  done


rm -r ${trav}

# soumet /home/demeyer/TRACKING/EULERIAN/1_PreProcessing/interp_MRCC6_to_MIROC6.sh -args UBE PN -jn interp_MRCC6_to_MIROC6_UBE_PN -listing /storm/demeyer/JOBS_OUTPUTS/ && soumet /home/demeyer/TRACKING/EULERIAN/1_PreProcessing/interp_MRCC6_to_MIROC6.sh -args UBE HU -jn interp_MRCC6_to_MIROC6_UBE_HU -listing /storm/demeyer/JOBS_OUTPUTS/ && soumet /home/demeyer/TRACKING/EULERIAN/1_PreProcessing/interp_MRCC6_to_MIROC6.sh -args UBF PN -jn interp_MRCC6_to_MIROC6_UBF_PN -listing /storm/demeyer/JOBS_OUTPUTS/ && soumet /home/demeyer/TRACKING/EULERIAN/1_PreProcessing/interp_MRCC6_to_MIROC6.sh -args UBF HU -jn interp_MRCC6_to_MIROC6_UBF_HU -listing /storm/demeyer/JOBS_OUTPUTS/ && soumet /home/demeyer/TRACKING/EULERIAN/1_PreProcessing/interp_MRCC6_to_MIROC6.sh -args UBD PN -jn interp_MRCC6_to_MIROC6_UBD_PN -listing /storm/demeyer/JOBS_OUTPUTS/ && soumet /home/demeyer/TRACKING/EULERIAN/1_PreProcessing/interp_MRCC6_to_MIROC6.sh -args UBD HU -jn interp_MRCC6_to_MIROC6_UBD_HU -listing /storm/demeyer/JOBS_OUTPUTS/