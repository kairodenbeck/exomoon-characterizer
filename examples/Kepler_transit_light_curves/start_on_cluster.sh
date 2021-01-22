#!/bin/bash


#conda activate run_with_openmp

export PATH=/home/rodenbeck/my_scratch/.conda/envs/run_with_openmp/bin:$PATH
export PYTHONPATH=/home/rodenbeck/Python_Modules/exomoondynamicfit:/home/rodenbeck/Python_Modules

#:/home/rodenbeck/my_scratch/exomoon-characterizer


PPN=$(cat $_CONDOR_JOB_AD | grep "RequestCpus =" | cut -f 3 -d " ")

echo PPN: $PPN


which mpiexec

if [ $PPN -eq 1 ]; then
    python generate_lightcurve.py $1 $2 $3
else
    mpiexec -np $PPN python generate_lightcurve.py $1 $2 $3
    #mpiexec -np $PPN  python ../run_mcmc_multiple_planets_stellar_density_automated.py $yaml_name use_MPI
fi

exit
