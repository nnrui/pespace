import json
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
fig_width_pt = 3*246.0                  
inches_per_pt = 1.0/72.27               
golden_mean = (np.sqrt(5)-1.0)/2.0      
fig_width = fig_width_pt*inches_per_pt  
fig_height = fig_width*golden_mean      
fig_size =  [fig_width,fig_height]
params = {'axes.labelsize': 20,
          'font.family': 'serif',
          'font.serif': 'Computer Modern Raman',
          'font.size': 18,
          'legend.fontsize': 20,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'axes.grid' : True,
          'text.usetex': True,
          'savefig.dpi' : 300,
          'lines.markersize' : 18,
          'figure.figsize': fig_size}
matplotlib.rcParams.update(params)

minimum_frequency = 1e-5
maximum_frequency = 1e-1
day = 3600*24
week = 7*day
month = 4*week
year =  12*month
duration = [day, week, month, year]
data_length_list = []
text_time = ['day', 'week', 'month', 'year']
for t in duration:
    f_array = np.arange(0, 1.0/(2*5), 1.0/t)
    bound = ((f_array >= minimum_frequency) * (f_array <= maximum_frequency))
    f_array = f_array[bound]
    data_length = len(f_array)
    data_length_list.append(data_length)

with open('time_consuming_gpu_tdi_response.json', 'r') as f:
    data_gpu = json.load(f)
with open('time_consuming_gpu_tdi_response_cpu.json', 'r') as f:
    data_cpu = json.load(f)

fig, ax = plt.subplots()
ax.loglog(data_cpu['data_length'], data_cpu['pespace'], label='pespace-cpu', linestyle='dashed', color='tab:blue' )
ax.loglog(data_cpu['data_length'], data_cpu['bbhx'],    label='bbhx-cpu',    linestyle='dashed', color='tab:orange')
ax.loglog(data_gpu['data_length'], data_gpu['pespace'], label='pespace-gpu', linestyle='solid', color='tab:blue')
ax.loglog(data_gpu['data_length'], data_gpu['bbhx'],    label='bbhx-gpu',    linestyle='solid', color='tab:orange')
ax.set_xlabel('data length')
ax.set_ylabel('time consuming (Sec)')
for idx, t in enumerate(data_length_list):
    ax.axvline(t, linestyle='dashed', color='tab:gray')
    ax.text(t, 10, text_time[idx])
ax.legend()
fig.savefig('perf_cpu_gpu.png')




with open('time_consuming_gpu_likelihood.json', 'r') as f:
    data_likelihood = json.load(f)

fig, ax = plt.subplots()
ax.loglog(data_likelihood['data_length'], data_likelihood['pespace'], label='pespace-gpu', linestyle='solid', color='tab:blue' )
ax.loglog(data_likelihood['data_length'], data_likelihood['bbhx'],    label='bbhx-cpu',    linestyle='dashed', color='tab:orange')
ax.set_xlabel('data length')
ax.set_ylabel('time consuming (Sec)')
for idx, t in enumerate(data_length_list):
    ax.axvline(t, linestyle='dashed', color='tab:gray')
    ax.text(t, 10, text_time[idx])
ax.legend()
fig.savefig('perf_likelihood.png')