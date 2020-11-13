#!/bin/bash
module purge
module load openmpi/gcc/1.10.2/64 gsl/2.4 fftw/gcc/openmpi-1.10.2/3.3.4 hdf5/gcc/1.8.16
cd /home/chainje
mpirun ./gizmo-public/GIZMO ./thesis/simulation/halos_only/sgr/gizmo.param
