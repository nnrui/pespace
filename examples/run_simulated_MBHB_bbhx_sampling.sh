#!/bin/sh
#SBATCH --job-name=LDC1-1_v1_MBHB_sampling
#SBATCH --chdir=/home/changfenggroup/nrui/works/codes/gw_space/pespace/examples
#SBATCH --output=/home/changfenggroup/nrui/works/codes/gw_space/pespace/examples/output/LDC1-1_v1_MBHB_bbhx_sampling.out
#SBATCH --partition=CPU-Shorttime
#SBATCH --nodes=5
#SBATCH --ntasks=200
#SBATCH --ntasks-per-node=40
#SBATCH --time=3-00
#SBATCH --qos=qos_cpu_shorttime

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# mpiexec -n 8 python _gpu_visible_test.py
mpiexec -n 200 python simulated_MBHB_bbhx_sampling.py

# #SBATCH --gpu-bind=verbose,map_gpu:0,1,2,3
# #SBATCH --gpu-bind=verbose,map_gpu:0,1,0,1
# #SBATCH --gpu-bind=verbose,single:1
# #SBATCH --gpu-bind=verbose,mask_gpu:0x1,0x2,0x1,0x2
# #SBATCH --gpu-bind=verbose,mask_gpu:0x1,0x2,0x4,0x8

# #SBATCH --cpus-per-gpu=1
# #SBATCH --ntasks-per-gpu=1

# #SBATCH --gres-flags=enforce-binding
# #SBATCH --partition=changfeng
