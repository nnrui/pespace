#!/bin/sh
#SBATCH --job-name perf_likelihood
#SBATCH --chdir /home/changfenggroup/nrui/works/codes/gw_space/pespace/ti_test
#SBATCH --output /home/changfenggroup/nrui/works/codes/gw_space/pespace/ti_test/perf_tdi_response_test_cpu.out
#SBATCH --partition changfeng 
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

python perf_tdi_response_test_cpu.py
