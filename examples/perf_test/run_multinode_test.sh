#!/bin/sh
#SBATCH --job-name=multinode_test
#SBATCH --chdir=/home/changfenggroup/nrui/works/codes/gw_space/pespace/examples/perf_test
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --gres=gpu:a100:8
#SBATCH --time=1-00
#SBATCH --qos=qos_a100_gpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# rm /home/changfenggroup/nrui/works/codes/gw_space/pespace/examples/output/multinode_test_output.dat
# for n in {1..80..5}; do
#     echo "running with ${n} processes"
#     mpiexec -n ${n} python multinode_test.py
# done
mpiexec -n 16 python multinode_test.py

#SBATCH --qos=gpujoblimit
