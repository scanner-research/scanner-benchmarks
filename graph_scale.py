import os
import subprocess
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks", {'axes.grid': True})

cpu_cinema_hist = [[10551.61, 2105.2, 1120.46, 962.30, 912.07, 836.48, 827.52]]
cpu_tvnews_hist = [[4218.9, 746.94, 469.67, 367.66, 429.78, 534.25, 341.49],
                   [4146.92, 763.20, 472.58, 407.31, 356.62, 330.53, 382.48],
                   [4281.43, 779.35, 482.42, 413.66, 349.45, 341.75, 379.63]]

#cpu_nodes = [10, 100, 300]
#cpu_nodes = [10, 50, 100, 150, 200, 250, 300]
cpu_nodes = [10, 50, 100, 150, 200, 250, 300]
cpu_linear = [n / cpu_nodes[0] for n in cpu_nodes]

cpu_single_hist = [[0, 0, 0]]
cpu_single_pose = [[0, 0, 0]]
cpu_nodes_single = [1, 10, 100]

gpu_cinema_hist = [[21767.48, 8740.74, 4388.72, 2951.43, 2232.26, 1762.29]]
gpu_cinema_pose = [[0, 0, 0, 0, 0, 0]]
gpu_tvnews_hist = [
    [5044.85, 1736.33, 964.19, 778.07, 573.83, 501.18],
    [5030.17, 1764.79, 961.05, 777.44, 584.21, 505.18],
    [5057.58, 1765.62, 976.25, 721.62, 595.63, 509.53]
    ]

gpu_tvnews_pose = [[0, 0, 0, 0, 0, 0]]

gpu_nodes = [10, 25, 50, 75, 100, 128]
gpu_linear = [n / gpu_nodes[0] for n in gpu_nodes]


gpu_single_hist = [[940.0, 110.2, 29.8]]
gpu_single_pose = [[0, 0, 0]]
gpu_nodes_single = [1, 10, 100]

def avg(ar):
    return [0 for _ in range(len(ar[0]))]

def speedup(ar):
    avg = [0 for _ in range(len(ar[0]))]
    for i in range(len(ar[0])):
        for j in range(len(ar)):
            avg[i] += ar[j][i]
        avg[i] /= len(ar)
    return [avg[0] / (n or 1) for n in avg]

scale = 2.5
w = 3.33 * scale 
h = 1.25 * scale
fig, (c_ax, g_ax) = plt.subplots(1, 2, figsize=(w, h))


# CPU graph
c_ax.set_title('Scaling using 32 core CPU Nodes on GCE')
c_ax.set_xlabel('# of nodes')
c_ax.set_ylabel('Speedup (relative to 10 nodes)')
c_ax.plot(cpu_nodes, speedup(cpu_cinema_hist), label='Cinema, Histogram')
c_ax.plot(cpu_nodes, speedup(cpu_tvnews_hist), label='TV News, Histogram')
c_ax.plot(cpu_nodes, cpu_linear, label='Linear', linestyle=':')
#c_ax.set_xlim(0, cpu_nodes[-1] * 1.1)
c_ax.set_xlim(0, 150)
c_ax.set_ylim(0, 12.5)
c_ax.legend()
sns.despine()

# GPU graph
g_ax.set_title('Scaling using 2xK80 Nodes on GCE')
g_ax.set_xlabel('# of nodes')
g_ax.set_ylabel('Speedup (relative to 10 nodes)')
g_ax.plot(gpu_nodes, speedup(gpu_cinema_hist), label='Cinema, Histogram')
g_ax.plot(gpu_nodes, speedup(gpu_cinema_pose), label='Cinema, OpenPose')
g_ax.plot(gpu_nodes, speedup(gpu_tvnews_hist), label='TV News, Histogram')
g_ax.plot(gpu_nodes, speedup(gpu_tvnews_pose), label='TV News, OpenPose')
g_ax.plot(gpu_nodes, gpu_linear, label='Linear', linestyle=':')
g_ax.set_xlim(0, gpu_nodes[-1] * 1.1)
g_ax.set_ylim(0, 12.5)
g_ax.legend()

fig.tight_layout()
sns.despine()

plt.savefig('batchscaling.png', dpi=300)
plt.savefig('batchscaling.pdf', dpi=300)
plt.clf()

# Total time
plt.title('Scaling using 32 core CPU Nodes on GCE')
plt.xlabel('# of nodes')
plt.ylabel('Total time')
plt.plot(cpu_nodes_single, avg(cpu_single_hist), label='Single Video, Histogram')
plt.xlim(0, cpu_nodes_single[-1] * 1.1)
plt.legend()
plt.show()
sns.despine()

plt.savefig('cpu_scale_single.png', dpi=300)
plt.clf()


# Total time
plt.title('Scaling using 1 K80 Nodes on GCE')
plt.xlabel('# of nodes')
plt.ylabel('Total time (seconds)')
plt.plot(gpu_nodes_single, avg(gpu_single_hist), label='Single Video, Histogram')
plt.plot(gpu_nodes_single, avg(gpu_single_pose), label='Single Video, OpenPose')
plt.xlim(0, cpu_nodes_single[-1] * 1.1)
plt.legend()
plt.show()
sns.despine()

plt.savefig('gpu_scale_single.png', dpi=300)
plt.clf()

