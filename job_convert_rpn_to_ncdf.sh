#!/bin/bash

#######################################################################
#  Script pour convertir les sorties RPN du MRCC6 en fichiers netcdf  #
#  de plusieurs simulations et variables à la suite en background     #
#######################################################################

# Liste des simulations
simu=(
    "ube"
    "ubf"
    "ubd"
)

# Liste des variables
vari=(
    "PN"
    "HU"
)

for s in "${simu[@]}"; do
    s_upper=$(echo "$s" | tr '[:lower:]' '[:upper:]')
    sim=$s
    arch_cdf="/storm/demeyer/TRACKING/EULERIAN/INPUTS/MRCC6_GEM5_$s_upper"
    arch_rpn="/cuyo/roberge/Output/GEM511/ARRIME/Cascades_CORDEX/CLASS/$s"
    for v in "${vari[@]}"; do
        var=$v
        bash /home/demeyer/TRACKING/EULERIAN/1_PreProcessing/convert_rpn_to_ncdf.sh "$sim" "$var" "$arch_cdf" "$arch_rpn"
    done
done

echo "Tous les jobs sont terminés."

#soumet /home/demeyer/TRACKING/EULERIAN/JOBS/job_convert_rpn_to_ncdf.sh -listing /home/demeyer/DATA/JOBS_OUTPUTS/