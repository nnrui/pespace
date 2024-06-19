#!/bin/sh
#SBATCH --job-name perf_likelihood
#SBATCH --chdir /home/changfenggroup/nrui/works/codes/gw_space/pespace/ti_test
#SBATCH --output /home/changfenggroup/nrui/works/codes/gw_space/pespace/ti_test/perf_likelihood_test.out
#SBATCH --partition changfeng 
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --gres=gpu:v100:1

python perf_likelihood_test.py
