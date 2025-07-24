#!/bin/bash
#SBATCH --job-name RETINASIM-HPC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=ret_%j.out
#SBATCH --error=ret_%j.err

# Activate conda environment
source /home/vamshis/miniconda3/bin/activate
conda activate retina
cd /home/vamshis/miniconda3/envs/retina/lib/python3.8/site-packages/retinasim

# Export below packages
export LD_LIBRARY_PATH=/home/apps/Compiler/cuda/cuda-11.2/nsight-systems-2020.3.2/host-linux-x64/Mesa:/home/vamshis/miniconda3/envs/retina/lib:$LD_LIBRARY_PATH
export PYTHONBREAKPOINT=0

# mpirun command 
time mpirun -np 1 python main.py /scratch/vamshis/FINAL/SIM_1/ --batch True --nbatch 30 --batch_offset 0 --simulate_injection True  
