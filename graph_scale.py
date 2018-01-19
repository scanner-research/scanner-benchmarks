import os
import subprocess
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['axes.titlesize'] = 6
mpl.rcParams['xtick.labelsize'] = 5
mpl.rcParams['ytick.labelsize'] = 5

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")
st = sns.axes_style()
st = {}
sns.set(style="ticks", rc={
    'axes.grid': True,
    'axes.linewidth': 0.5,
    'xtick.major.size': 2.0,
    'xtick.minor.size': 0.5,
    'ytick.major.size': 2.0,
    'ytick.minor.size': 0.5,
    'lines.linewidth': 0.5,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'font.size': 6,
})
#sns.set(st)


NAIVE_COLOR = '#504C48'
HIST_COLOR_A = '#ff6f69'
HIST_COLOR_B = '#d0b783'
POSE_COLOR_A = '#673888'
POSE_COLOR_B = '#c79dd7'

PEAK_COLOR = '#FF7169'
SPARK_COLOR = '#19838D'


labels_on = False

cpu_cinema_hist = [[10551.61, 2105.2, 1120.46, 962.30, 912.07, 836.48, 827.52]]
cpu_tvnews_hist = [[4218.9, 746.94, 469.67, 367.66, 429.78, 534.25, 341.49],
                   [4146.92, 763.20, 472.58, 407.31, 356.62, 330.53, 382.48],
                   [4281.43, 779.35, 482.42, 413.66, 349.45, 341.75, 379.63]]

#cpu_nodes = [10, 100, 300]
#cpu_nodes = [10, 50, 100, 150, 200, 250, 300]
cpu_nodes = [10, 50, 100, 150, 200, 250, 300]
cpu_linear = [n / cpu_nodes[0] for n in cpu_nodes]

cpu_single_hist = [[262.07, 58.488, 30.700, 25.63, 20.60]]
cpu_nodes_single = [1, 10, 25, 50, 75]

gpu_cinema_hist = [[21767.48, 8740.74, 4388.72, 2951.43, 2232.26, 1762.29]]
gpu_cinema_pose = [[10182.79, 3702.0, 1897.38, 1362.42, 1108.98, 933.50]]
gpu_tvnews_hist = [
    [5044.85, 1736.33, 964.19, 778.07, 573.83, 501.18],
    [5030.17, 1764.79, 961.05, 777.44, 584.21, 505.18],
    [5057.58, 1765.62, 976.25, 721.62, 595.63, 509.53]
    ]
gpu_tvnews_pose = [[13661.26, 3804.05, 1903.12, 1345.55, 1058.48, 962.86]]

gpu_nodes = [10, 25, 50, 75, 100, 128]
gpu_linear = [n / gpu_nodes[0] for n in gpu_nodes]


gpu_single_hist = [[947.5, 104.79, 56.28, 31.21, 27.26]]
gpu_single_pose = [[3318.3, 703.15, 313.13, 172.92, 123.99]]

gpu_nodes_single = [0.5, 5, 12, 25, 37]

cpu_cores = 32
gpu_cores = 2

def avg(ar):
    avg = [0 for _ in range(len(ar[0]))]
    for i in range(len(ar[0])):
        for j in range(len(ar)):
            avg[i] += ar[j][i]
        avg[i] /= len(ar)
    return avg

def speedup(ar):
    avg = [0 for _ in range(len(ar[0]))]
    for i in range(len(ar[0])):
        for j in range(len(ar)):
            avg[i] += ar[j][i]
        avg[i] /= len(ar)
    return [avg[0] / (n or 1) for n in avg]

scale = 1
w = 3.33 * scale
h = 1.25 * scale
fig, (c_ax, g_ax) = plt.subplots(1, 2, figsize=(w, h))

cpu_nodes = np.array(cpu_nodes) * cpu_cores

# CPU graph
c_ax.plot(cpu_nodes, speedup(cpu_cinema_hist), label='Cinema, Hist', color=HIST_COLOR_A)
c_ax.plot(cpu_nodes, speedup(cpu_tvnews_hist), label='TV News, Hist', color=HIST_COLOR_B)
c_ax.plot(cpu_nodes, cpu_linear, label='Linear', linestyle=':', color=NAIVE_COLOR)
#c_ax.set_xlim(0, cpu_nodes[-1] * 1.1)
c_ax.set_xlim(0, 150 * cpu_cores)
c_ax.set_ylim(0, 12.5)

if labels_on:
    c_ax.set_title('Scaling using 32 core CPU Nodes on GCE')
    c_ax.set_xlabel('# of CPU cores')
    c_ax.set_ylabel('Speedup (relative to 320 cores)')
    c_ax.legend()

sns.despine()

gpu_nodes = np.array(gpu_nodes) * gpu_cores

# GPU graph
g_ax.plot(gpu_nodes, speedup(gpu_cinema_hist), label='Cinema, Hist', color=HIST_COLOR_A)
g_ax.plot(gpu_nodes, speedup(gpu_cinema_pose), label='Cinema, Pose', color=POSE_COLOR_A)
g_ax.plot(gpu_nodes, speedup(gpu_tvnews_hist), label='TV News, Hist', color=HIST_COLOR_B)
g_ax.plot(gpu_nodes, speedup(gpu_tvnews_pose), label='TV News, Pose', color=POSE_COLOR_B)
g_ax.plot(gpu_nodes, gpu_linear, label='Linear', linestyle=':', color=NAIVE_COLOR)
g_ax.set_xlim(0, gpu_nodes[-1] * 1.1)
g_ax.set_ylim(0, 12.5)

if labels_on:
    g_ax.set_title('Scaling using 2xK80 Nodes on GCE')
    g_ax.set_xlabel('# of GPUs')
    g_ax.set_ylabel('Speedup (relative to 20 GPUs)')
    g_ax.legend()

fig.tight_layout()
sns.despine()

plt.savefig('batchscaling.png', dpi=300)
plt.savefig('batchscaling.pdf', dpi=300)
plt.clf()


scale = 1
w = 3.33 * scale
h = 1.25 * scale
fig, (c_ax, g_ax) = plt.subplots(1, 2, figsize=(w, h))

# Total time

cpu_nodes_single = np.array(cpu_nodes_single) * cpu_cores

c_ax.plot(cpu_nodes_single, avg(cpu_single_hist), label='Single Video, Histogram', color=HIST_COLOR_A)
c_ax.set_xlim(0, cpu_nodes_single[-1] * 1.1)
c_ax.set_ylim(0, 265)
c_ax.set_yticks([0, 30, 90, 150, 210, 270])

if labels_on:
    c_ax.set_title('Scaling using 32 core CPU Nodes on GCE')
    c_ax.set_xlabel('# of CPU cores')
    c_ax.set_ylabel('Total time (seconds)')
    c_ax.legend()

# Total time

gpu_nodes_single = np.array(gpu_nodes_single) * gpu_cores

g_ax.plot(gpu_nodes_single, avg(gpu_single_hist), label='Single Video, Hist', color=HIST_COLOR_A)
g_ax.plot(gpu_nodes_single, avg(gpu_single_pose), label='Single Video, Pose', color=POSE_COLOR_A)
g_ax.set_xlim(0, gpu_nodes_single[-1] * 1.1)
g_ax.set_ylim(0, 3500)
g_ax.set_yticks([0] + [x for x in range(500, 3501, 1000)])

if labels_on:
    g_ax.set_title('Scaling using 1 K80 Nodes on GCE')
    g_ax.set_xlabel('# of GPUs')
    g_ax.legend()

fig.tight_layout()
sns.despine()

plt.savefig('interactivescaling.png', dpi=300)
plt.savefig('interactivescaling.pdf', dpi=300)
plt.clf()
