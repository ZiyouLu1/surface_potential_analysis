#!/bin/bash
#SBATCH -J cpu_electron_density
#SBATCH --account ELLIS-SL3-CPU
#SBATCH --partition icelake
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total? (<= nodes*76)
#! The Ice Lake (icelake) nodes have 76 CPUs (cores) each and
#! 3380 MiB of memory per CPU.
#SBATCH --ntasks=76
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --mail-type=END

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-icl              # REQUIRED - loads the basic environment
module load python/3.11.0-icl

source ./venv/bin/activate
python3.11 ./src/copper_surface_analysis.py