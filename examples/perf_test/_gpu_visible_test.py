from mpi4py import MPI
import torch
import os

local_rank_num = int(os.environ['MPI_LOCALNRANKS'])
local_rank_id = int(os.environ['MPI_LOCALRANKID'])
gpus_list = os.environ['GPU_DEVICE_ORDINAL'].split(',')
selected_gpu = gpus_list[local_rank_id%len(gpus_list)]
os.environ['CUDA_VISIBLE_DEVICES'] = selected_gpu
comm = MPI.COMM_WORLD

gpu_num = torch.cuda.device_count()
visible_gpus = torch.cuda._parse_visible_devices()
current_gpu = torch.cuda.current_device()

print(f"MPI worker {comm.rank}/{comm.size}\n    visiable GPUs:\n    {gpu_num} with id {visible_gpus}\n    current device:\n    {current_gpu}")

# print('  SLURM_PROCID', os.environ['SLURM_PROCID'])
# print('  SLURM_LOCALID', os.environ['SLURM_LOCALID'])
# print('  SLURM_GTIDS', os.environ['SLURM_GTIDS'])
# print('  SLURM_JOB_ID', os.environ['SLURM_JOB_ID'])
# print('  MPI_LOCALNRANKS', os.environ['MPI_LOCALNRANKS'])
# print('  MPI_LOCALRANKID', os.environ['MPI_LOCALRANKID'])