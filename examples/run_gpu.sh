#!/bin/sh
#SBATCH --job-name=LISA_individual_MBHB
#SBATCH --chdir=/home/changfenggroup/nrui/works/codes/gw_space/pespace/examples
#SBATCH --partition=GPU-V100 
#SBATCH --nodes=2
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:v100:2
#SBATCH --time=10-00
#SBATCH --qos=gpujoblimit

# mpiexec -n 8 python _gpu_visible_test.py
mpiexec -n 24 python LISA_individual_MBHB.py


# #SBATCH --gpu-bind=verbose,map_gpu:0,1,2,3
# #SBATCH --gpu-bind=verbose,map_gpu:0,1,0,1
# #SBATCH --gpu-bind=verbose,single:1
# #SBATCH --gpu-bind=verbose,mask_gpu:0x1,0x2,0x1,0x2
# #SBATCH --gpu-bind=verbose,mask_gpu:0x1,0x2,0x4,0x8

# #SBATCH --cpus-per-gpu=1
# #SBATCH --ntasks-per-gpu=1

# #SBATCH --gres-flags=enforce-binding
