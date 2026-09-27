#!/bin/sh
#SBATCH --job-name=perf_test
#SBATCH --chdir=/home/changfenggroup/nrui/works/intro-ps/perf
#SBATCH --output=/home/changfenggroup/nrui/works/intro-ps/perf/%x_%j.out
#SBATCH --error=/home/changfenggroup/nrui/works/intro-ps/perf/%x_%j.err
#SBATCH --partition=GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=gnode16

set -e

PY=/home/changfenggroup/nrui/miniconda3/envs/pespace-clean/bin/python
run() {
    echo "=== [$(date '+%F %T')] $PY -u $1"
    "$PY" -u "$1" || { echo "!!! FAILED: $1"; exit 1; }
}

run perf_ll_bbhx_cpu.py
run perf_ll_bbhx_cuda.py
run perf_ll_ti_cpu.py
run perf_ll_ti_cuda.py
run perf_ll_jax_cpu.py
run perf_ll_jax_cuda.py

run perf_wf_lal.py
run perf_wf_ti_cpu.py
run perf_wf_ti_cuda.py
run perf_wf_jax_cpu.py
run perf_wf_jax_cuda.py
