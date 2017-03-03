#!/usr/bin/env python

from __future__ import print_function
import os
import os.path
import time
import sys
import struct
import json
from collections import defaultdict
from pprint import pprint
from datetime import datetime
import io
import csv
from collections import defaultdict as dd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['image.interpolation'] = 'nearest'
import seaborn as sns
from PIL import Image
import numpy as np

NAIVE_COLOR = '#21838c'
SCANNER_COLOR = '#F39948'
SCANNER2_COLOR = '#E83127'
PEAK_COLOR = '#FF7169'
SPARK_COLOR = '#19838D'

sns.set_style("ticks", {'axes.grid': True})


def get_trial_total_io_read(result):
    total_time, profilers = result

    total_io = 0
    # Per node
    for node, (_, profiler) in profilers.iteritems():
        for prof in profiler['load']:
            counters = prof['counters']
            total_io += counters['io_read'] if 'io_read' in counters else 0
    return total_io


def multi_gpu_graphs(test_name, frame_counts, frame_wh, results,
                     labels_on=True):
    #matplotlib.rcParams.update({'font.size': 22})
    scale = 2.5
    w = 3.33 * scale
    h = 1.25 * scale
    fig = plt.figure(figsize=(w, h))

    if False:
        fig.suptitle(
            "Scanner Multi-GPU Scaling on {width}x{height} video".format(
                width=frame_wh[test_name]['width'],
                height=frame_wh[test_name]['height'],
            ))
    ax = fig.add_subplot(111)
    if labels_on:
        ax.set_ylabel("Speedup (over 1 GPU)")
    ax.xaxis.grid(False)

    t = test_name
    # all_results = {'mean': {'caffe': [1074.2958445786187, 2167.455212488331, 4357.563170607772],
    #                                   'flow': [95.75056483734716, 126.96566457146966, 127.75415013154019],
    #                                   'histogram': [3283.778064650782,
    #                                                                         6490.032394321538,
    #                                                                         12302.865537345728]}}
    operations = [('histogram', 'HIST'),
                  ('caffe', 'DNN'),
                  ('flow', 'FLOW')]
    num_gpus = [1, 2, 4]

    ops = [op for op, _ in operations]
    labels = [l for _, l in operations]
    x = np.arange(len(labels)) * 1.2
    ys = [[0 for _ in range(len(num_gpus))] for _ in range(len(labels))]

    for j, op in enumerate(ops):
        for i, time in enumerate(results[t][op]):
            ys[j][i] = time

    for i in range(len(num_gpus)):
        xx = x + (i*0.35)
        fps = [ys[l][i] for l, _ in enumerate(labels)]
        y = [ys[l][i] / ys[l][0] for l, _ in enumerate(labels)]
        ax.bar(xx, y, 0.3, align='center', color=SCANNER_COLOR,
                    edgecolor='none')
        for (j, xy) in enumerate(zip(xx, y)):
            if i == 2:
                xyx = xy[0]
                xyy = xy[1] + 0.1
                if labels_on:
                    ax.annotate('{:d}'.format(int(fps[j])),
                                xy=(xyx, xyy), ha='center')
            if labels_on:
                ax.annotate("{:d}".format(num_gpus[i]), xy=(xy[0], -0.30),
                            ha='center', annotation_clip=False)

    yt = [0, 1, 2, 3, 4]
    ax.set_yticks(yt)
    ax.set_yticklabels(['{:d}'.format(d) for d in yt])
    ax.set_ylim([0, 4.2])

    ax.set_xticks(x+0.3)
    ax.set_xticklabels(labels, ha='center')
    fig.tight_layout()
    #ax.xaxis.labelpad = 10
    ax.tick_params(axis='x', which='major', pad=15)
    sns.despine()

    variants = ['1 GPU', '2 GPUs', '4 GPUs']

    name = 'multigpu_' + test_name
    fig.savefig(name + '.png', dpi=600)
    fig.savefig(name + '.pdf', dpi=600, transparent=True)
    with open(name + '_results.txt', 'w') as f:
        f.write('Speedup\n')
        f.write('{:10s}'.format(''))
        for l in variants:
            f.write('{:10s} |'.format(l))
        f.write('\n')
        for i, r in enumerate(ys):
            f.write('{:10s}'.format(labels[i]))
            for n in r:
                f.write('{:10f} |'.format(n / r[0]))
            f.write('\n')

        f.write('\nFPS\n')
        f.write('{:10s}'.format(''))
        for l in variants:
            f.write('{:10s} |'.format(l))
        f.write('\n')
        for i, r in enumerate(ys):
            f.write('{:10s}'.format(labels[i]))
            for n in r:
                f.write('{:10f} |'.format(n))
            f.write('\n')
    fig.clf()


def pose_reconstruction_graphs(results):
    results = []
    plt.clf()
    plt.cla()
    plt.close()

    sns.set_style('ticks')

    scale = 4.5
    #w = 3.33 * scale
    w = 1.6261 * scale
    h = 1.246 * scale
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)

    #plt.title("Scaling comparison for {}".format(pipeline))
    #ax.set_xlabel("Sampled / Base Time")
    #ax.set_ylabel("Accuracy (%)")
    ax.grid(b=False, axis='x')

    cams_list = [5, 10, 15, 20, 25, 30, 45, 60, 90, 120, 150, 180, 210, 240, 300, 360, 420, 480]
    times = [24.551988124847412, 25.52646803855896, 32.45369601249695, 32.8526508808136, 62.02082681655884, 63.67399311065674, 98.23960590362549, 86.65086603164673, 105.89570307731628, 139.70814990997314, 160.13016200065613, 183.25249886512756, 225.645281791687, 252.10664701461792, 307.928493976593, 403.465607881546, 501.95209407806396, 601.1797370910645]
    accuracy = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1245918367346939, 0.22112244897959168, 0.26448979591836735, 0.2866326530612246, 0.29724489795918385, 0.30755102040816334, 0.3137755102040817, 0.317857142857143, 0.3187755102040818, 0.3192857142857144], [0.20244897959183664, 0.4271428571428571, 0.5431632653061227, 0.5997959183673472, 0.6320408163265308, 0.6504081632653063, 0.6575510204081632, 0.6627551020408159, 0.6671428571428568, 0.6705102040816324], [0.32153061224489804, 0.6142857142857148, 0.7201020408163268, 0.7681632653061229, 0.799489795918367, 0.8161224489795917, 0.8325510204081632, 0.8401020408163267, 0.8480612244897958, 0.8572448979591835], [0.3009183673469387, 0.6352040816326532, 0.7828571428571425, 0.8359183673469388, 0.8756122448979592, 0.8906122448979591, 0.8973469387755104, 0.899795918367347, 0.9074489795918366, 0.9132653061224488], [0.3768367346938776, 0.7015306122448983, 0.8321428571428572, 0.8706122448979592, 0.8954081632653059, 0.907755102040816, 0.9156122448979593, 0.9173469387755102, 0.9223469387755102, 0.9273469387755107], [0.5231632653061223, 0.7931632653061221, 0.8673469387755101, 0.892448979591837, 0.9057142857142862, 0.9133673469387762, 0.9173469387755107, 0.9185714285714289, 0.918979591836735, 0.9194897959183678], [0.6140816326530617, 0.8255102040816327, 0.8821428571428571, 0.903469387755102, 0.9136734693877555, 0.9162244897959186, 0.9167346938775512, 0.917040816326531, 0.9176530612244902, 0.9177551020408167], [0.7026530612244903, 0.8578571428571428, 0.8913265306122449, 0.9046938775510205, 0.9115306122448981, 0.914693877551021, 0.9158163265306127, 0.9160204081632659, 0.9160204081632659, 0.9161224489795926], [0.7842857142857139, 0.8785714285714284, 0.9032653061224488, 0.9114285714285721, 0.9145918367346946, 0.9154081632653067, 0.9155102040816332, 0.9156122448979599, 0.9157142857142864, 0.9157142857142864], [0.8156122448979594, 0.8885714285714285, 0.9056122448979594, 0.9130612244897967, 0.9162244897959192, 0.9167346938775518, 0.9168367346938783, 0.9168367346938783, 0.9168367346938783, 0.9168367346938783], [0.8370408163265304, 0.8965306122448976, 0.9103061224489797, 0.9164285714285719, 0.919081632653062, 0.919285714285715, 0.9193877551020415, 0.9193877551020415, 0.9193877551020415, 0.9193877551020415], [0.8461224489795914, 0.9011224489795918, 0.9137755102040819, 0.9177551020408172, 0.9193877551020415, 0.919489795918368, 0.919489795918368, 0.919489795918368, 0.919489795918368, 0.919489795918368], [0.8596938775510203, 0.9010204081632657, 0.9158163265306127, 0.9187755102040822, 0.9202040816326535, 0.9202040816326535, 0.9203061224489801, 0.9204081632653066, 0.9204081632653066, 0.9204081632653066], [0.8735714285714283, 0.9056122448979592, 0.9165306122448986, 0.9198979591836739, 0.9214285714285718, 0.9215306122448983, 0.9216326530612249, 0.9216326530612249, 0.9216326530612249, 0.9216326530612249], [0.8981632653061224, 0.918979591836735, 0.9312244897959185, 0.9352040816326533, 0.9367346938775512, 0.9369387755102043, 0.9370408163265309, 0.9371428571428574, 0.9371428571428574, 0.9371428571428574], [0.936632653061225, 0.9511224489795919, 0.9562244897959188, 0.9587755102040822, 0.9602040816326536, 0.9603061224489801, 0.9603061224489801, 0.9603061224489801, 0.9603061224489801, 0.9603061224489801], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    relative_accuracy = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2200019068233354, 0.27652904040404025, 0.3046977169655742, 0.3150271335807051, 0.3237896310039169, 0.32781462585034027, 0.3314254792826222, 0.3362752525252525, 0.3407933931148216, 0.34228148835291683], [0.15539620696763548, 0.33249136775922494, 0.4272917182024324, 0.4857562358276645, 0.517957637600495, 0.5418097299525871, 0.554139713461142, 0.5633050144300144, 0.5712834982477838, 0.5785179344465059], [0.1784388785817357, 0.4091975147868003, 0.5199680200751627, 0.5769826502862219, 0.6055416309880597, 0.624973001205144, 0.6380683066933067, 0.649441253191253, 0.6611274677703248, 0.6710789448646591], [0.6832836726765299, 0.7546406569620858, 0.7854907552764695, 0.8007088268873982, 0.8108388258566832, 0.8190142793714225, 0.8235475794047221, 0.8258605719677145, 0.8296156640978067, 0.8321071864643294], [0.5453093434343433, 0.6737817747728461, 0.7152295462474031, 0.7341095491809785, 0.7436966347144918, 0.7512928777571635, 0.7602189051028337, 0.764250640232783, 0.768895679717108, 0.7761165808397947], [0.6649927314748745, 0.8000522077129217, 0.8391168295989726, 0.8529726483833625, 0.8620826415647842, 0.8666926426747859, 0.8713548594262882, 0.8746244172494173, 0.8791627598591886, 0.8824185219542361], [0.7539363790970934, 0.8752784992784993, 0.9141780560709133, 0.9307424757781901, 0.9404477942692238, 0.9461595547309838, 0.9489433621933624, 0.9502693516800662, 0.9513122294372297, 0.9523434601113174], [0.850369897959184, 0.9323852040816321, 0.9520025510204084, 0.9610331632653064, 0.9656505102040823, 0.9711352040816332, 0.9725000000000006, 0.9744260204081637, 0.9751147959183676, 0.9757780612244902], [0.921734693877551, 0.9733673469387761, 0.9877551020408175, 0.9941836734693884, 0.9958163265306126, 0.9965306122448981, 0.9971428571428572, 0.9972448979591837, 0.9973469387755103, 0.9973469387755103], [0.9433673469387758, 0.9840816326530623, 0.992755102040817, 0.9951020408163267, 0.9957142857142859, 0.9964285714285714, 0.9965306122448979, 0.9967346938775511, 0.9968367346938776, 0.9968367346938776], [0.9439795918367346, 0.9841836734693886, 0.9917346938775521, 0.9963265306122454, 0.9975510204081636, 0.9978571428571431, 0.9979591836734696, 0.9980612244897961, 0.9980612244897961, 0.9980612244897961], [0.9661224489795924, 0.9871428571428585, 0.993673469387756, 0.9969387755102045, 0.99765306122449, 0.9979591836734696, 0.9980612244897961, 0.9982653061224491, 0.9982653061224491, 0.9982653061224491], [0.9563265306122447, 0.9870408163265317, 0.9927551020408177, 0.9953061224489804, 0.9969387755102045, 0.9975510204081636, 0.99765306122449, 0.9980612244897961, 0.9980612244897961, 0.9980612244897961], [0.9606122448979595, 0.9784693877551028, 0.9814285714285722, 0.9824489795918374, 0.9834693877551026, 0.9838775510204089, 0.9841836734693884, 0.9842857142857149, 0.9842857142857149, 0.9842857142857149], [0.9529591836734701, 0.9653061224489804, 0.9738775510204094, 0.9758163265306135, 0.976122448979593, 0.9766326530612255, 0.9767346938775521, 0.9767346938775521, 0.9768367346938786, 0.9768367346938786], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    temporal_cams_list = [(30, 5), (60, 15), (90, 30), (120, 45), (150, 75), (200, 90), (240, 120)]
    temporal_times = [32.56036618550618, 43.92940409978231, 74.30371996561686, 110.34462833404541, 104.78588042259216, 119.68804597854614, 156.62090937296549]
    temporal_accuracy = [[0.045408163265306126, 0.08479591836734696, 0.12846938775510203, 0.16316326530612246, 0.18193877551020407, 0.19632653061224492, 0.2003061224489797, 0.20244897959183678, 0.20336734693877556, 0.20397959183673478], [0.20591836734693864, 0.47571428571428526, 0.6294897959183675, 0.7170408163265307, 0.772755102040816, 0.8006122448979588, 0.8128571428571424, 0.8239795918367341, 0.832857142857142, 0.8396938775510194], [0.3671428571428572, 0.6915306122448983, 0.8213265306122444, 0.8597959183673466, 0.8846938775510206, 0.89704081632653, 0.9051020408163266, 0.9068367346938776, 0.9115306122448981, 0.9165306122448981], [0.5187755102040817, 0.7906122448979588, 0.8640816326530608, 0.8889795918367348, 0.9022448979591836, 0.9093877551020411, 0.913163265306123, 0.9143877551020414, 0.9147959183673477, 0.9153061224489804], [0.658163265306123, 0.8352040816326527, 0.8817346938775512, 0.8991836734693875, 0.9087755102040821, 0.9136734693877561, 0.9151020408163272, 0.9155102040816334, 0.9156122448979599, 0.9157142857142864], [0.7041836734693883, 0.8591836734693876, 0.8921428571428571, 0.9050000000000001, 0.9119387755102042, 0.9148979591836741, 0.9158163265306127, 0.9160204081632659, 0.9160204081632659, 0.9161224489795926], [0.7842857142857139, 0.8785714285714284, 0.9033673469387754, 0.9113265306122454, 0.914489795918368, 0.9154081632653067, 0.9155102040816332, 0.9156122448979599, 0.9157142857142864, 0.9157142857142864]]

    total_cams = 480
    total_time = 4605

    x = []
    ys = [[] for _ in range(10)]
    for i in range(len(cams_list)):
        num_cams = cams_list[i]
        dnn_time = total_time * (num_cams / float(total_cams))
        time = dnn_time + times[i]
        x.append(time / (total_time + times[-1]))
        for j, acc in enumerate(accuracy[i]):
            ys[j].append(acc * 100)

    temporal_x = []
    temporal_ys = [[] for _ in range(10)]
    for i in range(len(temporal_cams_list)):
        major, minor = temporal_cams_list[i]
        num_cams = (major + minor * 14) / 15
        dnn_time = total_time * (num_cams / float(total_cams))
        time = dnn_time + times[i]
        temporal_x.append(time)
        for j, acc in enumerate(temporal_accuracy[i]):
            temporal_ys[j].append(acc * 100)

    # for y in ys:
    #     r = lambda: random.randint(0,255)
    #     c = '#%02X%02X%02X' % (r(),r(),r())
    #     ax.plot(x, y, color=c)
    ax.plot(x, ys[0], color=SCANNER_COLOR)
    ax.set_xlim([0, 1.0])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels([])
    ax.set_ylim([0, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([])
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    fig.tight_layout()
    sns.despine()
    # fig.legend(['Spark', 'Scanner CPU', 'Scanner GPU'], loc='upper left')
    test = '160422_mafia2'
    fig.savefig('poseReconstruction_{}.png'.format(test), dpi=300)
    fig.savefig('poseReconstruction_{}.pdf'.format(test), dpi=300)


def multi_node_graphs(all_results):
    all_results = json.loads("""
{"caffe_benchmark": {"1": {"scanner_CPU": 326.5170420000147,
                         "scanner_GPU": 387.8840370535301,
                         "spark": 27.543828161200317},
                     "2": {"scanner_CPU": 637.6501471607734,
                         "scanner_GPU": 748.3447519376416,
                         "spark": 54.10310186629111},
                     "4": {"scanner_CPU": 1274.8431035047988,
                         "scanner_GPU": 1498.6543110795326,
                         "spark": 97.36459048972931},
                     "8": {"scanner_CPU": 2503.757096120856,
                         "scanner_GPU": 2928.385396420333,
                         "spark": 125.10499842647118},
                     "16": {"scanner_CPU": 4719.057511025314,
                          "scanner_GPU": 5654.526329468247,
                          "spark": 279.3467864501258}},
"flow_benchmark": {"1": {"scanner_CPU": 22.54,
                        "scanner_GPU": 27.020355722139875,
                        "spark": 17.67208034395901},
                    "2": {"scanner_CPU": 43.04,
                        "scanner_GPU": 57.844015811703784,
                        "spark": 28.582607387713928},
                    "4": {"scanner_CPU": 88.05,
                        "scanner_GPU": 114.46687508997496,
                        "spark": 56.862139625037706},
                    "8": {"scanner_CPU": 173.92,
                        "scanner_GPU": 246.63184602361284,
                        "spark": 108.63880660528709},
                    "16": {"scanner_CPU": 330.19,
                         "scanner_GPU": 422.8535881899429,
                         "spark": 213.9296830948763}},
  "histogram_benchmark": {
    "1": {
      "spark": 1287.9036926392,
      "scanner_CPU": 990.0419956496697,
"scanner_GPU":623.0942113207435
    },
    "2": {
      "spark": 2346.2322872953,
      "scanner_CPU": 1843.3495521903167,
"scanner_GPU":1250.5331423300338
    },
    "4": {
      "spark":4016.8520829722156,
      "scanner_CPU":3880.2063559394037,
"scanner_GPU":2427.7997153326205
    },
    "8": {
      "spark":7366.958534590863,
      "scanner_CPU":7154.003787826941,
"scanner_GPU":5099.983276817758
    },
    "16": {
      "spark": 11843.088280945702,
      "scanner_CPU": 14433.693316025365,
"scanner_GPU":9999.63192837807
    }
  }}
""")

    def max_throughput_chart():
        plt.title("Spark vs. Scanner on 16 nodes")
        plt.xlabel("Pipeline")
        plt.ylabel("FPS")

        labels = pipelines
        x = np.arange(len(labels))
        ys = [[0, 0] for _ in range(len(labels))]

        for (i, pipeline) in enumerate(pipelines):
            values = all_results[pipeline]
            for (j, n) in enumerate(values.values()):
                ys[j][i] = n

        width = 0.3
        colors = sns.color_palette()
        for (i, y) in enumerate(ys):
            xx = x + (i*width)
            plt.bar(xx, y, width, align='center', color=colors[i])
            for (j, xy) in enumerate(zip(xx, y)):
                speedup = xy[1] / float(ys[0][j])
                plt.annotate('{} ({:.1f}x)'.format(int(xy[1]), speedup), xy=xy, ha='center')

        plt.xticks(x+width/2, labels)
        plt.legend(['Spark', 'Scanner'], loc='upper left')
        plt.tight_layout()

        plt.savefig('multinode.png', dpi=150)

    def scaling_chart():
        for pipeline in all_results:
            plt.clf()
            plt.cla()
            plt.close()

            scale = 2.5
            w = 1.75 * scale
            h = 1.25 * scale
            fig = plt.figure(figsize=(w, h))
            ax = fig.add_subplot(111)

            #plt.title("Scaling comparison for {}".format(pipeline))
            # ax.set_xlabel("Number of nodes")
            # ax.set_ylabel("FPS")
            ax.set_xticks(node_counts)
            if pipeline == 'histogram_benchmark':
                ax.set_yticks([4000, 8000, 12000, 16000])
                ax.set_yticklabels(['4k', '8k', '12k', '16k'])
                ax.set_ylim([0, 16000])
            elif pipeline == 'flow_benchmark':
                ax.set_yticks([150, 300, 450])
                ax.set_ylim([0, 450])
            elif pipeline == 'caffe_benchmark':
                ax.set_yticks([1500, 3000, 4500, 6000])
                ax.set_yticklabels(['1.5k', '3k', '4.5k', '6k'])
                ax.set_ylim([0, 6000])

            ax.grid(b=False, axis='x')

            x = node_counts
            values = all_results[pipeline]
            for (method, color) in [
                ('spark', SPARK_COLOR),
                ('scanner_CPU', SCANNER2_COLOR),
                ('scanner_GPU', SCANNER_COLOR)]:
                y = [values[str(n)][method] for n in node_counts]
                ax.plot(x, y, color=color)
                # for xy in zip(x,y):
                #     val = int(xy[1])
                #     speedup = xy[1] / values[str(xy[0])]['spark']
                #     if xy[0] == node_counts[0]:
                #         ha = 'left'
                #     elif xy[0] == node_counts[-1]:
                #         ha = 'right'
                #     else:
                #         ha = 'center'
                #     ax.annotate('{} ({:.1f}x)'.format(int(xy[1]), speedup), xy=xy, ha=ha)

            fig.tight_layout()
            sns.despine()
            # fig.legend(['Spark', 'Scanner CPU', 'Scanner GPU'], loc='upper left')
            fig.savefig('multinode_{}.png'.format(pipeline), dpi=150)
            fig.savefig('multinode_{}.pdf'.format(pipeline), dpi=150)

    scaling_chart()


def standalone_graphs(frame_counts, results):
    plt.clf()
    plt.title("Standalone perf on Charade")
    plt.ylabel("FPS")
    plt.xlabel("Pipeline")

    colors = sns.color_palette()

    x = np.arange(3)
    labels = ['caffe', 'flow', 'histogram']

    test_name = 'charade'
    tests = results[test_name]
    #for test_name, tests in results.iteritems():
    if 1:
        ys = []
        for i in range(len(tests[labels[0]])):
            y = []
            for label in labels:
                print(tests)
                frames = frame_counts[test_name]
                sec, timings = tests[label][i]
                if label == 'flow':
                    frames /= 20.0
                print(label, frames, sec, frames / sec)
                y.append(frames / sec)
            ys.append(y)

        print(ys)
        for (i, y) in enumerate(ys):
            xx = x+(i*0.3)
            plt.bar(xx, y, 0.3, align='center', color=colors[i])
            for xy in zip(xx, y):
                plt.annotate("{:.2f}".format(xy[1]), xy=xy)
                print(xy)
        plt.legend(['BMP', 'JPG', 'Video'], loc='upper left')
        plt.tight_layout()
        plt.savefig('standalone_' + test_name + '.png', dpi=150)
        plt.savefig('standalone_' + test_name + '.pdf', dpi=150)


def comparison_graphs(name,
                      width, height,
                      standalone_results, scanner_results,
                      peak_results,
                      labels_on=True):
    scale = 2.5
    w = 3.33 * scale
    h = 1.25 * scale
    fig = plt.figure(figsize=(w, h))
    if False:
        fig.suptitle("Microbenchmarks on {width}x{height} video".format(
            width=width,
            height=height))
    ax = fig.add_subplot(111)
    if labels_on:
        plt.ylabel("Speedup (over expert)")
    ax.xaxis.grid(False)

    # ops = ['histogram_cpu', 'histogram_gpu',
    #        'strided_hist_short_gpu', 'strided_hist_long_gpu',
    #        'range_hist_gpu',
    #        'flow_cpu', 'flow_gpu',
    #        'caffe']
    # labels = ['HISTCPU', 'HISTGPU',
    #           'SHORT', 'LONG',
    #           'RANGE',
    #           'FLOWCPU', 'FLOWGPU',
    #           'DNN']
    ops = ['histogram_cpu', 'histogram_gpu',
           'flow_cpu', 'flow_gpu',
           'caffe']
    labels = ['HISTCPU', 'HISTGPU',
              'FLOWCPU', 'FLOWGPU',
              'DNN']

    #for test_name, tests in results.iteritems():
    if 1:
        ys = []

        standalone_fps = []
        scanner_fps = []
        peak_fps = []

        standalone_y = []
        scanner_y = []
        peak_y = []
        for label in ops:
            v = peak_results[label][0]
            frames = v['frames']
            peak_sec, timings = v['results']
            p_fps = frames / peak_sec
            if peak_sec == -1:
                peak_fps.append(0)
            else:
                peak_fps.append(p_fps)
            peak_y.append(1.0)

            v = standalone_results[label][0]
            frames = v['frames']
            sec, timings = v['results']
            st_fps = frames / sec
            if sec == -1:
                standalone_y.append(0)
                standalone_fps.append(0)
            else:
                standalone_y.append(st_fps / p_fps)
                standalone_fps.append(st_fps)

            v = scanner_results[label][0]
            frames = v['frames']
            sec, timings = v['results']
            sc_fps = frames/sec
            if sec == -1:
                scanner_y.append(0)
                scanner_fps.append(0)
            else:
                scanner_y.append(sc_fps / p_fps)
                scanner_fps.append(sc_fps)

        fps = []
        fps.append(standalone_fps)
        fps.append(scanner_fps)
        fps.append(peak_fps)

        ys.append(standalone_y)
        ys.append(scanner_y)
        ys.append(peak_y)
        print(ys)

        x = np.arange(len(ops)) * 1.3

        variants = ['Baseline', 'Scanner', 'HandOpt']

        colors = [NAIVE_COLOR, SCANNER_COLOR, PEAK_COLOR]
        for (i, y) in enumerate(ys):
            xx = x+(i*0.35)
            ax.bar(xx, y, 0.3, align='center', color=colors[i],
                   edgecolor='none')
            if i == 1:
                for k, xxx in enumerate(xx):
                    ax.annotate("{}".format(labels[k]),
                                xy=(xxx, -0.08), annotation_clip=False,
                                ha='center')
            if i == 2:
                for k, xy in enumerate(zip(xx, y)):
                    xp, yp = xy
                    yp += 0.05
                    #xp += 0.1
                    ax.annotate("{:d}".format(int(peak_fps[k])), xy=(xp, yp),
                                ha='center')
        if False:
            plt.legend(['Non-expert', 'Scanner', 'Hand-authored'],
                       loc='upper right')

        ax.set_xticks(x+0.3)
        ax.set_xticklabels(['', '', ''])
        ax.xaxis.grid(False)

        yt = [0, 0.5, 1]
        ax.set_yticks(yt)
        ax.set_yticklabels(['{:.1f}'.format(d) for d in yt])
        ax.set_ylim([0, 1.1])


        plt.tight_layout()
        sns.despine()

        name = 'comparison_' + name
        plt.savefig(name + '.png', dpi=150)
        plt.savefig(name + '.pdf', dpi=150)
        with open(name + '_results.txt', 'w') as f:
            f.write('Speedup\n')
            f.write('{:10s}'.format(''))
            for l in variants:
                f.write('{:10s} |'.format(l))
            f.write('\n')
            for j in range(len(ys[0])):
                f.write('{:10s}'.format(labels[j]))
                for n in ys:
                    f.write('{:10f} |'.format(n[j]))
                f.write('\n')

            f.write('\nFPS\n')
            f.write('{:10s}'.format(''))
            for l in variants:
                f.write('{:10s} |'.format(l))
            f.write('\n')
            for j in range(len(fps[0])):
                f.write('{:10s}'.format(labels[j]))
                for n in fps:
                    f.write('{:10f} |'.format(n[j]))
                f.write('\n')
        plt.clf()


def striding_comparison_graphs(strides, results, labels_on=True):
    sns.set_style('ticks')
    scale = 2.5
    w = 3.33 * scale
    h = 1.25 * scale
    fig = plt.figure(figsize=(w, h))
    if False:
        fig.suptitle("Striding Comparison")
    ax = fig.add_subplot(111)
    if labels_on:
        plt.ylabel("Speedup (over no striding)")
    ax.xaxis.grid(False)

    colors = [NAIVE_COLOR, SCANNER_COLOR, SCANNER2_COLOR]
    #for test_name, tests in results.iteritems():
    labels = [name for name, _ in results]
    x = strides
    handles = []
    for (name, result), c in zip(results, colors):
        base_time = result['1'][0]['results'][0]
        y = []
        for stride in strides:
            time = result[str(stride)][0]['results'][0]
            y.append(base_time / time)
        h, = ax.plot(x, y, color=c)
        handles.append(h)

    #ax.plot(x, strides[1:], color=PEAK_COLOR)

    #ax.set_xlim([0, 1.0])
    ax.set_xticks(strides[1:])
    ax.set_xticklabels(strides[1:])
    #ax.set_ylim([0, strides[-1] * 1.1])
    #ax.set_yticks(strides)
    #ax.set_yticklabels(strides)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    fig.tight_layout()
    sns.despine()
    fig.legend(handles, labels, loc='upper right')
    name = 'striding_comparison'
    fig.savefig('{:s}.png'.format(name), dpi=300)
    fig.savefig('{:s}.pdf'.format(name), dpi=300)

    plt.clf()


def convert_time(d):
    def convert(t):
        return '{:2f}'.format(t / 1.0e9)
    return {k: convert_time(v) if isinstance(v, dict) else convert(v) \
            for (k, v) in d.iteritems()}

def generate_statistics(profilers):
    totals = {}
    for _, profiler in profilers.values():
        for kind in profiler:
            if not kind in totals: totals[kind] = {}
            for thread in profiler[kind]:
                for (key, start, end) in thread['intervals']:
                    if not key in totals[kind]: totals[kind][key] = 0
                    totals[kind][key] += end-start

    readable_totals = convert_time(totals)
    return readable_totals


def graph_io_rate_benchmark(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    mb = 0
    wis_per_node = defaultdict(list)
    for row in rows:
        wis = int(row['work_item_size'])
        lwpn = row['load_workers_per_node']
        mbs = row['MB/s']
        embs = row['Effective MB/s']
        mb = row['MB']
        wis_per_node[wis].append([lwpn, mbs, embs])

    wis = [64, 128, 256, 512, 1024, 2048, 4096, 8096]
    colors = ['g', 'b', 'k', 'w', 'm', 'c', 'r', 'y']
    plt.clf()
    ax = plt.subplot(1, 1, 1)
    for w, c in zip(wis, colors):
        d = wis_per_node[w]
        print(d)
        ax.plot(map(lambda x: x[0], d),
                map(lambda x: x[1], d),
                color=c,
                linestyle='--')
        ax.plot(map(lambda x: x[0], d),
                map(lambda x: x[2], d),
                color=c,
                linestyle='-',
                label=str(w) + ' wis')

    ax.set_xlabel('Load threads')
    ax.set_ylabel('MB/s')
    ax.legend()

    #ax.set_title('Loading ' + mb + ' MB on bodega SSD')
    #plt.savefig('io_rate_bodega.png', dpi=150)
    ax.set_title('Loading ' + mb + ' MB on GCS')
    plt.savefig('io_rate_gcs.png', dpi=150)


def graph_decode_rate_benchmark(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    wis_per_node = defaultdict(list)
    for row in rows:
        print(row)
        wis = int(row['work_item_size'])
        pus = int(row['pus_per_node'])
        t = float(row['time'])
        df = int(row['decoded_frames'])
        ef = int(row['effective_frames'])
        wis_per_node[wis].append([pus, t, df, ef])

    #wis = [64, 128, 256, 512, 1024, 2048]
    wis = [128, 256, 512, 1024, 2048, 4096]
    colors = ['g', 'b', 'k', 'y', 'm', 'c', 'r', 'w']
    plt.clf()
    ax = plt.subplot(1, 1, 1)
    for w, c in zip(wis, colors):
        d = wis_per_node[w]
        ax.plot(map(lambda x: x[0], d),
                map(lambda x: x[2]/x[1], d),
                color=c,
                linestyle='--')
        ax.plot(map(lambda x: x[0], d),
                map(lambda x: x[3]/x[1], d),
                color=c,
                linestyle='-',
                label=str(w) + ' wis')

    ax.set_xlabel('PUs')
    ax.set_ylabel('Decode FPS')
    ax.legend()

    ax.set_title('Decoding frames on Intel')
    plt.savefig('decode_rate_intel.png', dpi=150)


def multi_node_max_throughput_chart(all_results):
    pipelines = [
        # 'histogram_benchmark',
       'caffe_benchmark',
        # 'flow_benchmark'
    ]

    plt.title("Spark vs. Scanner on 16 nodes")
    plt.xlabel("Pipeline")
    plt.ylabel("FPS")

    labels = pipelines
    x = np.arange(len(labels))
    ys = [[0, 0] for _ in range(len(labels))]

    for (i, pipeline) in enumerate(pipelines):
        values = all_results[pipeline]
        for (j, n) in enumerate(values.values()):
            ys[j][i] = n

    width = 0.3
    colors = sns.color_palette()
    for (i, y) in enumerate(ys):
        xx = x + (i*width)
        plt.bar(xx, y, width, align='center', color=colors[i])
        for (j, xy) in enumerate(zip(xx, y)):
            speedup = xy[1] / float(ys[0][j])
            plt.annotate('{} ({:.1f}x)'.format(int(xy[1]), speedup), xy=xy, ha='center')

    plt.xticks(x+width/2, labels)
    plt.legend(['Spark', 'Scanner'], loc='upper left')
    plt.tight_layout()

    plt.savefig('multinode.png', dpi=150)


def multi_node_scaling_chart(all_results):
    for pipeline in all_results:
        plt.clf()
        plt.cla()
        plt.close()

        scale = 2.5
        w = 1.75 * scale
        h = 1.25 * scale
        fig = plt.figure(figsize=(w, h))
        ax = fig.add_subplot(111)

        #plt.title("Scaling comparison for {}".format(pipeline))
        # ax.set_xlabel("Number of nodes")
        # ax.set_ylabel("FPS")
        ax.set_xticks(node_counts)
        if pipeline == 'histogram_benchmark':
            ax.set_yticks([4000, 8000, 12000, 16000])
            ax.set_yticklabels(['4k', '8k', '12k', '16k'])
            ax.set_ylim([0, 16000])
        elif pipeline == 'flow_benchmark':
            ax.set_yticks([150, 300, 450])
            ax.set_ylim([0, 450])
        elif pipeline == 'caffe_benchmark':
            ax.set_yticks([1500, 3000, 4500, 6000])
            ax.set_yticklabels(['1.5k', '3k', '4.5k', '6k'])
            ax.set_ylim([0, 6000])

        ax.grid(b=False, axis='x')

        x = node_counts
        values = all_results[pipeline]
        for (method, color) in [
            ('spark', SPARK_COLOR),
            ('scanner_CPU', SCANNER2_COLOR),
            ('scanner_GPU', SCANNER_COLOR)]:
            y = [values[str(n)][method] for n in node_counts]
            ax.plot(x, y, color=color)
            # for xy in zip(x,y):
            #     val = int(xy[1])
            #     speedup = xy[1] / values[str(xy[0])]['spark']
            #     if xy[0] == node_counts[0]:
            #         ha = 'left'
            #     elif xy[0] == node_counts[-1]:
            #         ha = 'right'
            #     else:
            #         ha = 'center'
            #     ax.annotate('{} ({:.1f}x)'.format(int(xy[1]), speedup), xy=xy, ha=ha)

        fig.tight_layout()
        sns.despine()
        # fig.legend(['Spark', 'Scanner CPU', 'Scanner GPU'], loc='upper left')
        fig.savefig('multinode_{}.png'.format(pipeline), dpi=150)
        fig.savefig('multinode_{}.pdf'.format(pipeline), dpi=150)


def graph_main(args):
    graph_decode_rate_benchmark('decode_test.csv')
