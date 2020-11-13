#!/bin/bash
#SBATCH --job-name=mw_init      # create a short name for your job
#SBATCH --nodes=1               # node count
#SBATCH --exclusive             # fully dedicate single node 
<<<<<<< HEAD
#SBATCH -p physics
=======
#SBATCH -p all
>>>>>>> 7307f7c4702a96cbeb178b29e96673b9f162f825
#SBATCH --ntasks-per-node=25
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=50G               # memory for entire job
#SBATCH --time=24:00:00         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all         # send email on job start, end, and fail
#SBATCH --mail-user=chainje@princeton.edu
module purge
module load openmpi/gcc/1.10.2/64 gsl/2.4 fftw/gcc/openmpi-1.10.2/3.3.4 hdf5/gcc/1.8.16
cd /home/chainje
srun ./gizmo-public/GIZMO ./thesis/simulation/halos_only/mw/gizmo.param
