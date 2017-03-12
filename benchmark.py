from __future__ import print_function
import os
import os.path
import time
import subprocess
import sys
import struct
from collections import defaultdict
from pprint import pprint
from datetime import datetime
from collections import defaultdict as dd
import tempfile
from multiprocessing import cpu_count
import toml
from PIL import Image
from timeit import default_timer as now
import string
import random
import glob
from scannerpy import Database, DeviceType, ScannerException
from scannerpy.stdlib import NetDescriptor
from scannerpy.config import Config
import numpy as np
from pprint import pprint
import graph
import copy

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCANNER_DIR = os.path.join(SCRIPT_DIR, '..', 'scanner')
COMPARISON_DIR = os.path.join(SCRIPT_DIR, 'comparison')
DEVNULL = open(os.devnull, 'wb', 0)


def clear_filesystem_cache():
    os.system('sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"')


def make_db(opts):
    config_path = opts['config_path'] if 'config_path' in opts else None
    db_path = opts['db_path'] if 'db_path' in opts else None
    config = Config(config_path)
    if db_path is not None:
        config.db_path = db_path
    return Database(master='localhost:5001', workers=['localhost:5003'], config=config)


def run_trial(tasks, pipeline, collection_name, opts={}):
    print('Running trial: collection {:s} ...:'.format(collection_name))
    # Clear cache
    config_path = opts['config_path'] if 'config_path' in opts else None
    db_path = opts['db_path'] if 'db_path' in opts else None
    config = Config(config_path)
    if db_path is not None:
        config.db_path = db_path
    workers = ['localhost']
    if 'nodes' in opts:
        workers = opts['nodes']
    #db = Database(master='localhost:5001', workers=workers, config=config)
    db = Database(master='localhost:5001', workers=['localhost:5003'], config=config)
    scanner_opts = {}
    def add_opt(s):
        if s in opts:
            scanner_opts[s] = opts[s]
    add_opt('work_item_size')
    add_opt('cpu_pool')
    add_opt('gpu_pool')
    add_opt('pipeline_instances_per_node')

    start = now()
    success = True
    prof = None
    try:
        clear_filesystem_cache()
        out_collection = db.run(tasks, pipeline, collection_name, force=True,
                                **scanner_opts)
        prof = out_collection.profiler()
        elapsed = now() - start
        total = prof.total_time_interval()
        t = (total[1] - total[0])
        t /= float(1e9)  # ns to s
        print('Trial succeeded: {:.3f}s elapsed, {:.3f}s effective'.format(
            elapsed, t))
    except ScannerException as e:
        elapsed = now() - start
        success = False
        prof = None
        print('Trial FAILED after {:.3f}s: {:s}'.format(elapsed, str(e)))
        t = -1
    db.stop_cluster()
    return success, t, prof


def run_opencv_trial(video_file, gpus_per_node, batch_size):
    print('Running opencv trial: {:d} gpus, {:d} batch size'.format(
        gpus_per_node,
        batch_size
    ))
    clear_filesystem_cache()
    current_env = os.environ.copy()
    start = time.time()
    program_path = os.path.join(
        COMPARISON_DIR, 'build/opencv/opencv_compare')
    p = subprocess.Popen([
        program_path,
        '--video_paths_file', video_file,
        '--gpus_per_node', str(gpus_per_node),
        '--batch_size', str(batch_size)
    ], env=current_env, stdout=DEVNULL, stderr=subprocess.STDOUT)
    pid, rc, ru = os.wait4(p.pid, 0)
    elapsed = time.time() - start
    if rc != 0:
        print('Trial FAILED after {:.3f}s'.format(elapsed))
        elapsed = -1
    else:
        print('Trial succeeded, took {:.3f}s'.format(elapsed))
    return elapsed


def run_caffe_trial(net, net_descriptor_file, device_type, net_input_width,
                    net_input_height, num_elements, batch_size):
    print(('Running trial: {}, {}, {}, {:d}x{:d} net input, {:d} elements, '
           '{:d} batch_size').format(
               net,
               net_descriptor_file,
               device_type,
               net_input_width,
               net_input_height,
               num_elements,
               batch_size
           ))
    clear_filesystem_cache()
    current_env = os.environ.copy()
    if device_type == "CPU":
        current_env["OMP_NUM_THREADS"] = "68"
        current_env["KMP_BLOCKTIME"] = "10000000"
        current_env["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
    start = time.time()
    p = subprocess.Popen([
        'build/comparison/caffe/caffe_throughput',
        '--net_descriptor_file', net_descriptor_file,
        '--device_type', device_type,
        '--net_input_width', str(net_input_width),
        '--net_input_height', str(net_input_height),
        '--num_elements', str(num_elements),
        '--batch_size', str(batch_size),
    ], env=current_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ''.join([line for line in p.stdout])
    pid, rc, ru = os.wait4(p.pid, 0)
    elapsed = time.time() - start
    profiler_output = {}
    if rc != 0:
        print('Trial FAILED after {:.3f}s'.format(elapsed))
        print(output, file=sys.stderr)
        # elapsed = -1
    else:
        print('Trial succeeded, took {:.3f}s'.format(elapsed))
        elapsed *= float(1000)  # s to ms
    return elapsed


def print_trial_times(title, trial_settings, trial_times):
    print(' {:^58s} '.format(title))
    print(' =========================================================== ')
    print(' Nodes | GPUs/n | Batch | Loaders | Total Time | Eval Time ')
    for settings, t in zip(trial_settings, trial_times):
        total_time = t[0]
        eval_time = 0
        for prof in t[1]['eval']:
            for interval in prof['intervals']:
                if interval[0] == 'task':
                    eval_time += interval[2] - interval[1]
        eval_time /= float(len(t[1]['eval']))
        eval_time /= float(1000000000)  # ns to s
        print(' {:>5d} | {:>6d} | {:>5d} | {:>5d} | {:>9.3f}s | {:>9.3f}s '
              .format(
                  settings['node_count'],
                  settings['gpus_per_node'],
                  settings['batch_size'],
                  settings['load_workers_per_node'],
                  total_time,
                  eval_time))


def print_opencv_trial_times(title, trial_settings, trial_times):
    print(' {:^58s} '.format(title))
    print(' =========================================================== ')
    print(' Nodes | GPUs/n | Batch | Loaders | Total Time ')
    for settings, t in zip(trial_settings, trial_times):
        total_time = t[0]
        print(' {:>5d} | {:>6d} | {:>5d} | {:>5d} | {:>9.3f}s '
              .format(
                  1,
                  settings['gpus_per_node'],
                  settings['batch_size'],
                  1,
                  total_time))


def print_caffe_trial_times(title, trial_settings, trial_times):
    print(' {:^58s} '.format(title))
    print(' ================================================================= ')
    print(' Net      | Device |    WxH    | Elems | Batch | Time   | ms/frame ')
    for settings, times in zip(trial_settings, trial_times):
        total_time = min([t for t in times if t != -1])
        print((' {:>8s} | {:>6s} | {:>4d}x{:<4d} | {:>5d} | {:>5d} | {:>6.3f}s '
               ' {:>8.3f}ms')
              .format(
                  settings['net'],
                  settings['device_type'],
                  settings['net_input_width'],
                  settings['net_input_height'],
                  settings['num_elements'],
                  settings['batch_size'],
                  total_time / 1000.0,
                  total_time / settings['num_elements']))


def dicts_to_csv(headers, dicts):
    output = io.BytesIO()
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    for d in dicts:
        writer.writerow(d)
    return output.getvalue()


def get_trial_total_io_read(result):
    total_time, profilers = result

    total_io = 0
    # Per node
    for node, (_, profiler) in profilers.iteritems():
        for prof in profiler['load']:
            counters = prof['counters']
            total_io += counters['io_read'] if 'io_read' in counters else 0
    return total_io


def video_encoding_benchmark():
    input_video = '/bigdata/wcrichto/videos/charade_short.mkv'
    num_frames = 2878 # TODO(wcrichto): automate this
    output_video = '/tmp/test.mkv'
    video_paths = '/tmp/videos.txt'
    dataset_name = 'video_encoding'
    input_width = 1920
    input_height = 1080

    variables = {
        'scale': {
            'default': 0,
            'range': []
        },
        'crf': {
            'default': 23,
            'range': [1, 10, 20, 30, 40, 50]
        },
        'gop': {
            'default': 25,
            'range': [5, 15, 25, 35, 45]
        }
    }

    pipelines = [
        'effective_decode_rate',
        'histogram',
        'knn_patches'
    ]

    variables['scale']['default'] = '{}x{}'.format(input_width, input_height)
    for scale in [1, 2, 3, 4, 8]:
        width = input_width / scale
        height = input_height / scale
        # FFMPEG says dimensions must be multiple of 2
        variables['scale']['range'].append('{}x{}'.format(width//2 * 2,
                                                          height//2 * 2))

    command_template = """
ffmpeg -i {input} -vf scale={scale} -c:v libx264 -x264opts \
    keyint={gop}:min-keyint={gop} -crf {crf} {output}
"""

    scanner_settings = {
        'force': True,
        'node_count': 1,
        'pus_per_node': 1,
        'work_item_size': 512
    }

    all_results = {}
    for pipeline in pipelines:
        all_results[pipeline] = {}
        for var in variables:
            all_results[pipeline][var] = {}

    for current_var in variables:
        settings = {'input': input_video, 'output': output_video}
        for var in variables:
            settings[var] = variables[var]['default']

        var_range = variables[current_var]['range']
        for val in var_range:
            settings[current_var] = val
            os.system('rm -f {}'.format(output_video))
            cmd = command_template.format(**settings)
            if os.system(cmd) != 0:
                print('Error: bad ffmpeg command')
                print(cmd)
                exit()

            result, _ = db.ingest('video', dataset_name, [output_video], {'force': True})
            if result != True:
                print('Error: failed to ingest')
                exit()

            for pipeline in pipelines:
                _, result = run_trial(dataset_name, pipeline, 'test',
                                      scanner_settings)
                stats = generate_statistics(result)
                if pipeline == 'effective_decode_rate':
                    t = stats['eval']['decode']
                elif pipeline == 'histogram':
                    t = float(stats['eval']['evaluate']) - \
                        float(stats['eval']['decode'])
                else:
                    t = float(stats['eval']['caffe:net']) + \
                        float(stats['eval']['caffe:transform_input'])

                fps = '{:.3f}'.format(num_frames / float(t))
                all_results[pipeline][current_var][val] = fps

    pprint(all_results)


def count_frames(video):
    cmd = """
    ffprobe -v error -count_frames -select_streams v:0 \
          -show_entries stream=nb_read_frames \
          -of default=nokey=1:noprint_wrappers=1 \
           {}
    """
    return int(subprocess.check_output(cmd.format(video), shell=True))


def multi_gpu_benchmark(tests, frame_counts, frame_wh):
    db_path = '/tmp/scanner_multi_gpu_db'

    db = scanner.Scanner()
    scanner_settings = {
        'db_path': db_path,
        'node_count': 1,
        'pus_per_node': 1,
        'io_item_size': 256,
        'work_item_size': 64,
        'tasks_in_queue_per_pu': 3,
        'force': True,
        'env': {
            'SC_JOB_NAME': 'base'
        }
    }
    dataset_name = 'multi_gpu'
    video_job = 'base'

    #num_gpus = [1]
    #num_gpus = [4]
    num_gpus = [1, 2, 4]
    #num_gpus = [2, 4]
    operations = [('histogram', 'histogram_benchmark'),
                  #('caffe', 'caffe_benchmark')]
                  ('caffe', 'caffe_benchmark'),
                  ('flow', 'flow_benchmark')]


    all_results = {}
    for test_name, paths in tests.iteritems():
        all_results[test_name] = {}
        for op, _ in operations:
            all_results[test_name][op] = []

        #frames = count_frames(video)
        os.system('rm -rf {}'.format(db_path))
        print('Ingesting {}'.format(paths))
        # ingest data
        db = make_db(scanner_settings)
        result, _ = db.ingest('video', dataset_name, paths, scanner_settings)
        if result is False:
            print('Failed to ingest')
            exit()

        scanner_settings['env']['SC_JOB_NAME'] = video_job

        for op, pipeline in operations:
            for gpus in num_gpus:
                frames = frame_counts[test_name]
                if op == 'histogram':
                    if frame_wh[test_name]['width'] == 640:
                        scanner_settings['io_item_size'] = 2048
                        scanner_settings['work_item_size'] = 1024
                    else:
                        scanner_settings['io_item_size'] = 512
                        scanner_settings['work_item_size'] = 128
                elif op == 'flow':
                    frames /= 20
                    scanner_settings['io_item_size'] = 512
                    scanner_settings['work_item_size'] = 64
                elif op == 'caffe':
                    scanner_settings['io_item_size'] = 480
                    scanner_settings['work_item_size'] = 96
                elif op == 'caffe_cpm2':
                    scanner_settings['io_item_size'] = 256
                    scanner_settings['work_item_size'] = 64

                scanner_settings['node_count'] = gpus
                print('Running {}, {} GPUS'.format(op, gpus))
                t, _ = run_trial(dataset_name, pipeline,
                                 op, scanner_settings)
                print(t, frames / float(t))
                all_results[test_name][op].append(float(frames) / float(t))

    pprint(all_results)
    return all_results


def run_cmd(template, settings):
    cmd = template.format(**settings)
    if os.system(cmd) != 0:
        print('Bad command: {}'.format(cmd))
        exit()


def multi_node_benchmark():
    dataset_name = 'multi_node_benchmark2'
    spark_dir = '/users/wcrichto/spark-2.1.0-bin-hadoop2.7'

    videos_dir = '/users/wcrichto/videos'
    videos = [
        'fightClub_clip.mp4',
        # 'anewhope.m4v',
        # 'brazil.mkv',
        # 'fightClub.mp4',
        # 'excalibur.mp4'
    ]

    # videos_dir = '/users/wcrichto'
    # with open('dfouhey_videos.txt') as f:
    #     videos = [s.strip() for s in f.read().split("\n")][:1000]

    pipelines = [
        # 'histogram_benchmark',
       'caffe_benchmark',
        # 'flow_benchmark'
    ]

    node_counts = [1, 2, 4, 8, 16]
    hosts = ['h{}.sparkcluster.blguest.orca.pdl.cmu.edu'.format(i) for i in range(node_counts[-1])]

#     db = scanner.Scanner()
#     scanner_settings = {
#         'force': True,
#         'hosts': hosts,
#         'tasks_in_queue_per_pu': 2,
#         'env': {
#             'SC_JOB_NAME': 'base'
#         }
#     }

#     # print('Ingesting...')
#     # result, _ = db.ingest(
#     #     'video',
#     #     dataset_name,
#     #     ['{}/{}'.format(videos_dir, video) for video in videos],
#     #     {'force': True})
#     # if result is False:
#     #     print('Failed to ingest')
#     #     exit()

#     run_spark = '{spark_dir}/run_sparkcaffe.sh {node_count} {pipeline} {input_dir}'
#     split_video = """
# ffmpeg -i {videos_dir}/{input} -vcodec copy -acodec copy -segment_time 60 \
#   -f segment {videos_dir}/segments/{segment_dir}/segment%03d.mp4
# """

#     #total_frames = 304677
#     #total_frames = 138000 * 3
#     #total_frames = 200352 * 2
#     total_frames = 129600

#     # print('Splitting...')
#     # os.system('rm -rf {}/segments/*'.format(videos_dir))
#     # for video in videos:
#     #     segment_dir = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
#     #     os.system('mkdir -p {}/segments/{}'.format(videos_dir, segment_dir))
#     #     run_cmd(split_video, {
#     #         'input': video,
#     #         'segment_dir': segment_dir,
#     #         'videos_dir': videos_dir
#     #     })
#     #     # total_frames += count_frames('{}/{}'.format(videos_dir, video))

#     # print('Total frames', total_frames)

#     def append(result):
#         with open("progress", "a") as f:
#             f.write(result + "\n")

#     all_results = {}
#     for pipeline in pipelines:
#         all_results[pipeline] = {}
#         for node_count in node_counts:
#             all_results[pipeline][node_count] = {}

#             scanner_settings['node_count'] = node_count

#             if pipeline == 'histogram_benchmark':
#                 configs = [
#                     ('GPU', 1, 4096, 1024),
#                     ('CPU', 8, 1024, 128)
#                     ]
#             elif pipeline == 'flow_benchmark':
#                 configs = [
#                     # ('GPU', 1, 96, 96),
#                     ('CPU', 16, 16, 16)
#                     ]
#             else:
#                 configs = [
#                     ('GPU', 1, 512, 256),
#                     # ('CPU', 1, 1024, 512)
#                     ]


#             for (device, pus, io_item_size, work_item_size) in configs:
#                 scanner_settings['env']['SC_DEVICE'] = device
#                 scanner_settings['pus_per_node'] = pus
#                 scanner_settings['io_item_size'] = io_item_size
#                 scanner_settings['work_item_size'] = work_item_size
#                 t, _ = run_trial(dataset_name, pipeline, 'test', scanner_settings)
#                 all_results[pipeline][node_count]['scanner_{}'.format(device)] = \
#                   total_frames / t
#                 append(json.dumps(all_results))

#             # start = now()
#             # if pipeline == 'caffe_benchmark':
#             #     spark_node_count = node_count # GPUs
#             # else:
#             #     spark_node_count = node_count * 16 # CPUs
#             # run_cmd(run_spark, {
#             #     'node_count': spark_node_count,
#             #     'spark_dir': spark_dir,
#             #     'pipeline': pipeline,
#             #     'input_dir': '{}/segments'.format(videos_dir)
#             # })
#             # t = now() - start
#             # all_results[pipeline][node_count]['spark'] = total_frames / t
#             # append(json.dumps(all_results))

#     pprint(all_results)
#     exit()

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



def image_video_decode_benchmark():
    input_video = '/bigdata/wcrichto/videos/charade_short.mkv'
    num_frames = 2878 # TODO(wcrichto): automate this
    output_video = '/tmp/test.mkv'
    output_im_bmp = '/tmp/test_bmp'
    output_im_jpg = '/tmp/test_jpg'
    paths_file = '/tmp/paths.txt'
    dataset_name = 'video_encoding'
    in_job_name = scanner.Scanner.base_job_name()
    input_width = 1920
    input_height = 1080

    scales = []

    for scale in [1, 2, 3, 4, 8]:
        width = input_width / scale
        height = input_height / scale
        # FFMPEG says dimensions must be multiple of 2
        scales.append('{}x{}'.format(width//2 * 2, height//2 * 2))

    scale_template = "ffmpeg -i {input} -vf scale={scale} -c:v libx264 {output}"
    jpg_template = "ffmpeg -i {input} {output}/frame%07d.jpg"
    bmp_template = "ffmpeg -i {input} {output}/frame%07d.bmp"

    db = scanner.Scanner()
    scanner_settings = {
        'force': True,
        'node_count': 1,
        'work_item_size': 512
    }

    def run_cmd(template, settings):
        cmd = template.format(**settings)
        if os.system(cmd) != 0:
            print('Bad command: {}'.format(cmd))
            exit()

    all_results = {}
    for scale in scales:
        all_results[scale] = {}

        os.system('rm {}'.format(output_video))
        run_cmd(scale_template, {
            'input': input_video,
            'output': output_video,
            'scale': scale
        })

        os.system('mkdir -p {path} && rm -f {path}/*'.format(path=output_im_bmp))
        run_cmd(bmp_template, {
            'input': output_video,
            'output': output_im_bmp
        })

        os.system('mkdir -p {path} && rm -f {path}/*'.format(path=output_im_jpg))
        run_cmd(jpg_template, {
            'input': output_video,
            'output': output_im_jpg
        })

        datasets = [('video', [output_video], 'effective_decode_rate'),
                    ('image', ['{}/{}'.format(output_im_bmp, f)
                               for f in os.listdir(output_im_bmp)],
                     'image_decode_rate'),
                    ('image', ['{}/{}'.format(output_im_jpg, f)
                               for f in os.listdir(output_im_jpg)],
                     'image_decode_rate')]

        for (i, (ty, paths, pipeline)) in enumerate(datasets):
            result, _ = db.ingest(ty, dataset_name, paths, {'force': True})
            if result != True:
                print('Error: failed to ingest')
                exit()

            pus_per_node = cpu_count() if pipeline == 'image_decode_rate' else 1
            scanner_settings['pus_per_node'] = pus_per_node
            t, result = run_trial(dataset_name, in_job_name, pipeline,
                                  'test', scanner_settings)
            stats = generate_statistics(result)
            all_results[scale][i] = {
                'decode': stats['eval']['decode'],
                'io': stats['load']['io'],
                'total': t
            }

    pprint(all_results)


def disk_size(path):
    output = subprocess.check_output("du -bh {}".format(path), shell=True)
    return output.split("\t")[0]


def storage_benchmark():
    config_path = '/tmp/scanner.toml'
    output_video = '/tmp/test.mkv'
    output_video_stride = '/tmp/test_stride.mkv'
    output_images_jpg = '/tmp/test_jpg'
    output_images_bmp = '/tmp/test_bmp'
    output_images_stride = '/tmp/test_jpg_stride'
    paths_file = '/tmp/paths.txt'
    dataset_name = 'video_encoding'

    video_paths = {
        'charade': '/bigdata/wcrichto/videos/charade_short.mkv',
        'meangirls': '/bigdata/wcrichto/videos/meanGirls_medium.mp4'
    }

    datasets = [(video, scale)
                for video in [('charade', 1920, 1080, 2878),
                              ('meangirls', 640, 480, 5755)]
                for scale in [1, 2, 4, 8]]

    strides = [1, 2, 4, 8]
    disks = {
        'sdd': '/data/wcrichto/db',
        'hdd': '/bigdata/wcrichto/db',
    }

    scale_template = "ffmpeg -i {input} -vf scale={scale} -c:v libx264 {output}"
    jpg_template = "ffmpeg -i {input} {output}/frame%07d.jpg"
    bmp_template = "ffmpeg -i {input} {output}/frame%07d.bmp"
    stride_template = "ffmpeg -f image2 -i {input}/frame%*.jpg {output}"

    scanner_settings = {
        'force': True,
        'node_count': 1,
        'work_item_size': 96,
        'pus_per_node': 1,
        'config_path': config_path
    }

    scanner_toml = scanner.ScannerConfig.default_config_path()
    with open(scanner_toml, 'r') as f:
        scanner_config = toml.loads(f.read())

    all_results = []
    all_sizes = []
    for ((video, width, height, num_frames), scale) in datasets:
        width /= scale
        height /= scale
        scale = '{}x{}'.format(width//2*2, height//2*2)

        os.system('rm -f {}'.format(output_video))
        run_cmd(scale_template, {
            'input': video_paths[video],
            'scale': scale,
            'output': output_video
        })

        os.system('mkdir -p {path} && rm -f {path}/*'.format(path=output_images_jpg))
        run_cmd(jpg_template, {
            'input': output_video,
            'output': output_images_jpg
        })

        os.system('mkdir -p {path} && rm -f {path}/*'.format(path=output_images_bmp))
        run_cmd(bmp_template, {
            'input': output_video,
            'output': output_images_bmp
        })

        for stride in strides:
            os.system('mkdir -p {path} && rm -f {path}/*'
                      .format(path=output_images_stride))
            for frame in range(0, num_frames, stride):
                os.system('ln -s {}/frame{:07d}.jpg {}'
                          .format(output_images_jpg, frame, output_images_stride))
            os.system('rm -f {}'.format(output_video_stride))
            run_cmd(stride_template, {
                'input': output_images_stride,
                'output': output_video_stride
            })

            jobs = [
                ('orig_video', 'video', [output_video], 'effective_decode_rate', stride),
                ('strided_video', 'video', [output_video_stride], 'effective_decode_rate', 1),
                ('exploded_jpg', 'image',
                 ['{}/{}'.format(output_images_jpg, f)
                  for f in os.listdir(output_images_jpg)],
                 'image_decode_rate', stride),
                ('exploded_bmp', 'image',
                 ['{}/{}'.format(output_images_bmp, f)
                  for f in os.listdir(output_images_bmp)],
                 'image_decode_rate', stride)
            ]

            config = (video, scale, stride)
            all_sizes.append((config, {
                'orig_video': disk_size(output_video),
                'strided_video': disk_size(output_video_stride),
                'exploded_jpg': disk_size(output_images_jpg),
                'exploded_bmp': disk_size(output_images_bmp)
            }))

            for disk in disks:
                scanner_config['storage']['db_path'] = disks[disk]
                with open(config_path, 'w') as f:
                    f.write(toml.dumps(scanner_config))
                db = scanner.Scanner(config_path=config_path)
                for (job_label, ty, paths, pipeline, pipeline_stride) in jobs:
                    config = (video, scale, stride, disk, job_label)
                    print('Running test: ', config)
                    result, _ = db.ingest(ty, dataset_name, paths, {'force': True})
                    if result != True:
                        print('Error: failed to ingest')
                        exit()

                    with open('stride.txt', 'w') as f:
                        f.write(str(pipeline_stride))
                    t, result = run_trial(dataset_name, in_job_name, pipeline,
                                          'test', scanner_settings)
                    stats = generate_statistics(result)
                    all_results.append((config, {
                        'decode': stats['eval']['decode'],
                        'io': stats['load']['io'],
                        'total': t
                    }))

    print(json.dumps(all_results))
    print(json.dumps(all_sizes))


def standalone_benchmark(video, video_frames, tests):
    output_dir = '/tmp/standalone'
    test_output_dir = '/tmp/standalone_outputs'
    paths_file = os.path.join(output_dir, 'paths.txt')

    def read_meta(path):
        files = [name for name in os.listdir(path)
                 if os.path.isfile(os.path.join(path, name))]
        filename = os.path.join(path, files[0])
        with Image.open(filename) as im:
            width, height = im.size
        return {'num_images': len(files) - 2, 'width': width, 'height': height}

    def write_meta_file(path, meta):
        with open(os.path.join(path, 'meta.txt'), 'w') as f:
            f.write(str(meta['num_images']) + '\n')
            f.write(str(meta['width']) + '\n')
            f.write(str(meta['height']))

    def write_paths(paths):
        with open(paths_file, 'w') as f:
            f.write(paths[0])
            for p in paths[1:]:
                f.write('\n' + p)

    def run_standalone_trial(input_type, paths_file, operation, frames, stride):
        print('Running standalone trial: {}, {}, {}'.format(
            input_type,
            paths_file,
            operation))
        clear_filesystem_cache()
        current_env = os.environ.copy()
        start = time.time()
        program_path = os.path.join(
            COMPARISON_DIR, 'build/standalone/standalone_comparison')
        p = subprocess.Popen([
            program_path,
            '--input_type', input_type,
            '--paths_file', paths_file,
            '--operation', operation,
            '--frames', str(frames),
            '--stride', str(stride),
        ], env=current_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        so, se = p.communicate()
        rc = p.returncode
        elapsed = time.time() - start
        timings = {}
        if rc != 0:
            print('Trial FAILED after {:.3f}s'.format(elapsed))
            print(so)
            elapsed = -1
        else:
            print('Trial succeeded, took {:.3f}s'.format(elapsed))
            for line in so.splitlines():
                if line.startswith('TIMING: '):
                    k, s, v = line[len('TIMING: '):].partition(",")
                    timings[k] = float(v)
            elapsed = timings['total']
        return elapsed, timings

    bmp_template = "ffmpeg -i {input} -start_number 0 {output}/frame%07d.bmp"
    jpg_template = "ffmpeg -i {input} -start_number 0 {output}/frame%07d.jpg"

    operations = {
        'histogram_cpu': 'histogram_cpu',
        'histogram_gpu': 'histogram_gpu',
        'flow_cpu': 'flow_cpu',
        'flow_gpu': 'flow_gpu',
        'caffe': 'caffe',
        'gather_hist_cpu': 'histogram_cpu',
        'gather_hist_gpu': 'histogram_gpu',
    }

    os.system('rm -rf {}'.format(output_dir))
    os.system('mkdir {}'.format(output_dir))

    run_paths = []
    base = os.path.basename(video)
    run_path = os.path.join(output_dir, base)
    os.system('cp {} {}'.format(video, run_path))
    run_paths.append(run_path)
    write_paths(run_paths)

    results = {}
    for test in tests:
        name = test['name']

        test_type = None
        for o, tt in operations.iteritems():
            if name.startswith(o):
                test_type = tt
                break
        assert(test_type is not None)

        sampling = test['sampling']
        results[name] = []

        frames = video_frames
        stride = 1
        sampling_type = sampling[0]
        if sampling_type == 'all':
            frames = video_frames
        elif sampling_type == 'range':
            assert(len(sampling[1]) == 1)
            assert(sampling[1][0][0] == 0)
            frames = sampling[1][0][1]
        elif sampling_type == 'strided':
            frames = video_frames
            stride = sampling[1]

        os.system('rm -rf {}'.format(test_output_dir))
        os.system('mkdir -p {}'.format(test_output_dir))
        total, timings = run_standalone_trial('mp4', paths_file, test_type,
                                              frames, stride)
        results[name].append({'results': (total, timings),
                              'frames': frames / stride})

    print(results)
    return results


def scanner_benchmark(video, total_frames, tests):
    db_dir = '/tmp/scanner_db'
    default_scanner_settings = {
        'db_path': db_dir,
        'node_count': 1,
        'work_item_size': 96,
        'tasks_in_queue_per_pu': 3,
        'force': True,
        'env': {}
    }
    db = make_db(default_scanner_settings)
    collection_name = 'test'

    # Histogram benchmarks
    hist_cpu = db.ops.Histogram(device=DeviceType.CPU)
    hist_gpu = db.ops.Histogram(device=DeviceType.GPU)

    # Optical flow benchmarks
    def of_pipeline(dt):
        table_input = db.ops.Input()
        of = db.ops.OpticalFlow(
            inputs=[(table_input, ["frame", "frame_info"])],
            device=dt)
        disc = db.ops.Discard(
            inputs=[(of, ["flow"])],
            device=dt)
        return disc

    of_cpu = of_pipeline(DeviceType.CPU)
    of_gpu = of_pipeline(DeviceType.GPU)

    # Caffe benchmark
    descriptor = NetDescriptor.from_file(db, 'nets/googlenet.toml')
    caffe_args = db.protobufs.CaffeArgs()
    caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())
    caffe_args.batch_size = 96

    def caffe_pipeline(device_type):
        table_input = db.ops.Input()
        caffe_input = db.ops.CaffeInput(
            inputs=[(table_input, ["frame", "frame_info"])],
            args=caffe_args,
            device=device_type)
        caffe = db.ops.Caffe(
            inputs=[(caffe_input, ["caffe_frame"]),
                    (table_input, ["frame_info"])],
            args=caffe_args,
            device=device_type)
        return caffe

    caffe_cpu = caffe_pipeline(DeviceType.CPU)
    caffe_gpu = caffe_pipeline(DeviceType.GPU)

    # Multi-view stereo benchmark

    operations = {
        'histogram_cpu': hist_cpu,
        'histogram_gpu': hist_gpu,
        'flow_cpu': of_cpu,
        'flow_gpu': of_gpu,
        'caffe': caffe_gpu,
        'strided_hist_cpu': hist_cpu,
        'strided_hist_gpu': hist_gpu,
        'strided_hist_short_gpu': hist_gpu,
        'strided_hist_long_gpu': hist_gpu,
        'gather_hist_cpu': hist_cpu,
        'gather_hist_gpu': hist_gpu,
        'range_hist_cpu': hist_cpu,
        'range_hist_gpu': hist_gpu,
    }

    os.system('rm -rf {}'.format(db_dir))

    # ingest data
    db.stop_cluster()
    db = make_db(default_scanner_settings)
    collection, f = db.ingest_video_collection(collection_name, [video],
                                               force=True)
    db.stop_cluster()
    assert(len(f) == 0)

    results = {}
    for test in tests:
        name = test['name']

        ops = None
        for o, pipeline in operations.iteritems():
            if name.startswith(o):
                ops = pipeline
                break
        assert(ops is not None)

        settings = test['scanner_settings']
        sampling = test['sampling']

        results[name] = []

        # Parse sampling
        item_size = settings['item_size']
        sampling_type = sampling[0]
        print(sampling)
        if sampling_type == 'all':
            sampled_input = db.sampler().all(collection, item_size=item_size)
            frames = total_frames
        elif sampling_type == 'strided':
            sampled_input = db.sampler().strided(collection,
                                                 sampling[1],
                                                 item_size=item_size)
            frames = total_frames / sampling[1]
        elif sampling_type == 'gather':
            sampled_input = [db.sampler().gather(collection,
                                                 sampling[1],
                                                 item_size=item_size)]
            frames = len(sampling[1])
        elif sampling_type == 'range':
            sampled_input = db.sampler().ranges(collection,
                                                sampling[1],
                                                item_size=item_size)
            frames = sum(e - s for s, e in sampling[1])
        else:
            print('Not a valid sampling type:', sampling_type)
            exit(1)

        # Parse settings
        opts = default_scanner_settings.copy()
        opts.update(settings)
        opts['work_item_size'] = 96
        print('Running {:s}'.format(name))
        #frame_factor = 50
        success, total, prof = run_trial(sampled_input, ops, 'out', opts)
        assert(success)
        stats = prof.statistics()
        prof.write_trace(name + '.trace')
        results[name].append({'results': (total, stats), 'frames': frames})

    print(video)
    pprint(results)
    return results


def peak_benchmark(tests, frame_counts, wh):
    db_dir = '/tmp/scanner_db'
    test_output_dir = '/tmp/peak_outputs'

    db = scanner.Scanner()
    db._db_path = db_dir
    scanner_settings = {
        'db_path': db_dir,
        'node_count': 1,
        'pus_per_node': 1,
        'io_item_size': 256,
        'work_item_size': 64,
        'tasks_in_queue_per_pu': 3,
        'force': True,
        'env': {}
    }
    dataset_name = 'test'
    video_job = 'base'

    def run_peak_trial(list_path, op, width, height, decoders, evaluators):
        print('Running peak trial: {}'.format(op))
        clear_filesystem_cache()
        current_env = os.environ.copy()
        start = time.time()
        program_path = os.path.join(
            COMPARISON_DIR, 'build/peak/peak_comparison')
        p = subprocess.Popen([
            program_path,
            '--video_list_path', list_path,
            '--operation', op,
            '--decoder_count', str(decoders),
            '--eval_count', str(evaluators),
            '--width', str(width),
            '--height', str(height),
        ], env=current_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        so, se = p.communicate()
        rc = p.returncode
        elapsed = time.time() - start
        timings = {}
        if rc != 0:
            print('Trial FAILED after {:.3f}s'.format(elapsed))
            print(so)
            elapsed = -1
        else:
            print('Trial succeeded, took {:.3f}s'.format(elapsed))
            for line in so.splitlines():
                if line.startswith('TIMING: '):
                    k, s, v = line[len('TIMING: '):].partition(",")
                    timings[k] = float(v)
            elapsed = timings['total']
        return elapsed, timings

    split_video = """ffmpeg {time} -i {input} -vcodec copy -acodec copy -segment_time {seg} -f segment {videos_dir}/segments/{segment_dir}/segment%03d.mp4"""

    def split_videos(videos, videos_dir, time, seg):
        print('Splitting...')
        os.system('rm -rf {}/segments/*'.format(videos_dir))

        paths = []
        for video in videos:
            segment_dir = ''.join(random.choice(string.ascii_uppercase +
                                                string.digits)
                                  for _ in range(8))
            os.system('mkdir -p {}/segments/{}'.format(videos_dir,
                                                       segment_dir))
            run_cmd(split_video, {
                'input': video,
                'segment_dir': segment_dir,
                'videos_dir': videos_dir,
                'time': str(time),
                'seg': seg
            })
            files = glob.glob('{}/segments/{}/*'.format(
                videos_dir, segment_dir))
            print(files)
            for f in files:
                paths.append(f)
            #total_frames += count_frames('{}/{}'.format(videos_dir, video))

        #print('Total frames', total_frames)
        print('num paths', len(paths))
        return paths


    operations = [('histogram_cpu', 16, 16, '', '180'),
                  ('histogram_gpu', 1, 1, '', '180'),
                  ('flow_cpu', 1, 32, '-ss 00:00:00 -t 00:06:00', '5'),
                  ('flow_gpu', 1, 1, '-ss 00:00:00 -t 00:06:00', '5'),
                  ('caffe', 1, 1, '', '180')]

    all_results = {}
    for test_name, paths in tests.iteritems():
        all_results[test_name] = {}
        for op, _, _, _, _ in operations:
            all_results[test_name][op] = []

        # video
        for op, dec, ev, tt, seg in operations:
            os.system('rm -rf {}'.format(test_output_dir))
            os.system('mkdir -p {}'.format(test_output_dir))
            frames = frame_counts[test_name]
            if op == 'flow_cpu' or op == 'flow_gpu':
                frames = 8632

            # ingest data
            video_paths = split_videos(paths, '/tmp/peak_videos', tt, seg)
            os.system('rm /tmp/peak_videos.txt')
            with open('/tmp/peak_videos.txt', 'w') as f:
                for p in video_paths:
                    f.write(p + '\n')

            all_results[test_name][op].append(
                {'results': run_peak_trial('/tmp/peak_videos.txt', op,
                                           wh[test_name]['width'],
                                           wh[test_name]['height'], dec, ev),
                 'frames': frames})

    print(all_results)
    return all_results


def scanner_striding_test(video, total_frames, tests):
    db_dir = '/tmp/scanner_db'
    default_scanner_settings = {
        'db_path': db_dir,
        'node_count': 1,
        'work_item_size': 64,
        'tasks_in_queue_per_pu': 3,
        'force': True,
        'env': {}
    }
    db = make_db(default_scanner_settings)
    collection_name = 'test'

    # Histogram benchmarks
    hist_gpu = db.ops.Histogram(device=DeviceType.GPU)

    os.system('rm -rf {}'.format(db_dir))

    # ingest data
    db.stop_cluster()
    db = make_db(default_scanner_settings)
    collection, f = db.ingest_video_collection(collection_name, [video],
                                               force=True)
    db.stop_cluster()
    assert(len(f) == 0)

    results = {}
    for test in tests:
        name = test['name']

        ops = hist_gpu
        settings = test['scanner_settings']
        sampling = test['sampling']

        results[name] = []

        # Parse sampling
        item_size = settings['item_size']
        sampling_type = sampling[0]
        print(sampling)
        if sampling_type == 'all':
            sampled_input = db.sampler().all(collection, item_size=item_size)
            frames = total_frames
        elif sampling_type == 'strided':
            sampled_input = db.sampler().strided(collection,
                                                 sampling[1],
                                                 item_size=item_size)
            frames = total_frames / sampling[1]
        elif sampling_type == 'gather':
            sampled_input = [db.sampler().gather(collection,
                                                 sampling[1],
                                                 item_size=item_size)]
            frames = len(sampling[1])
        elif sampling_type == 'range':
            sampled_input = db.sampler().ranges(collection,
                                                sampling[1],
                                                item_size=item_size)
            frames = sum(e - s for s, e in sampling[1])
        else:
            print('Not a valid sampling type:', sampling_type)
            exit(1)

        # Parse settings
        opts = default_scanner_settings.copy()
        opts.update(settings)
        opts['work_item_size'] = 64
        print('Running {:s}'.format(name))
        #frame_factor = 50
        success, total, prof = run_trial(sampled_input, ops, 'out', opts)
        assert(success)
        stats = prof.statistics()
        prof.write_trace(name + '.trace')
        results[name].append({'results': (total, stats), 'frames': frames})

    print(video)
    pprint(results)
    return results


def decode_sol(tests, frame_count):
    db_dir = '/tmp/scanner_db'
    input_video = '/tmp/scanner_db/datasets/test/data/0_data.bin'

    db = scanner.Scanner()
    db._db_path = db_dir
    scanner_settings = {
        'db_path': db_dir,
        'node_count': 1,
        'pus_per_node': 1,
        'io_item_size': 8192,
        'work_item_size': 4096,
        'tasks_in_queue_per_pu': 4,
        'force': True,
        'env': {}
    }
    dataset_name = 'test'
    video_job = 'base'

    decode_pipeline = 'effective_decode_rate'

    def run_ocv_trial(ty, path):
        print('Running ocv trial: {}'.format(path))
        clear_filesystem_cache()
        current_env = os.environ.copy()
        start = time.time()
        program_path = os.path.join(
            COMPARISON_DIR, 'build/ocv_decode/ocv_decode')
        p = subprocess.Popen([
            program_path,
            '--decoder', ty,
            '--path', path,
        ], env=current_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        so, se = p.communicate()
        rc = p.returncode
        elapsed = time.time() - start
        timings = {}
        if rc != 0:
            print('Trial FAILED after {:.3f}s'.format(elapsed))
            print(so)
            elapsed = -1
        else:
            print('Trial succeeded, took {:.3f}s'.format(elapsed))
            for line in so.splitlines():
                if line.startswith('TIMING: '):
                    k, s, v = line[len('TIMING: '):].partition(",")
                    timings[k] = float(v)
            elapsed = timings['total']
        return elapsed, timings

    def run_cmd(template, settings):
        cmd = template.format(**settings)
        if os.system(cmd) != 0:
            print('Bad command: {}'.format(cmd))
            exit()

    ffmpeg_cpu_template = 'ffmpeg -vcodec h264 -i {path} -f null -'
    ffmpeg_gpu_template = 'ffmpeg -vcodec h264_cuvid -i {path} -f null -'


    all_results = {}
    for test_name, paths in tests.iteritems():
        assert(len(paths) == 1)
        path = paths[0]

        all_results[test_name] = {}

        vid_path = '/tmp/vid'

        os.system('rm -rf {}'.format(db_dir))
        os.system('cp {} {}'.format(path, vid_path))

        # ingest data
        result, _ = db.ingest('video', dataset_name, paths, scanner_settings)
        assert(result)

        if test_name == 'mean':
            scanner_settings['io_item_size'] = 8192
            scanner_settings['work_item_size'] = 2048
        if test_name == 'fight':
            scanner_settings['io_item_size'] = 2048
            scanner_settings['work_item_size'] = 512


        # Scanner decode
        total, prof = run_trial(dataset_name, decode_pipeline, 'test',
                                scanner_settings)
        all_results[test_name]['scanner'] = total

        # OCV decode
        total, _ = run_ocv_trial('cpu', vid_path)
        all_results[test_name]['opencv_cpu'] = total

        total, _ = run_ocv_trial('gpu', vid_path)
        all_results[test_name]['opencv_gpu'] = total

        # FFMPEG CPU decode
        start_time = time.time()
        run_cmd(ffmpeg_cpu_template, {'path': vid_path})
        all_results[test_name]['ffmpeg_cpu'] = time.time() - start_time

        # FFMPEG GPU decode
        start_time = time.time()
        run_cmd(ffmpeg_gpu_template, {'path': vid_path})
        all_results[test_name]['ffmpeg_gpu'] = time.time() - start_time

        print('Decode test on ', test_name)
        print("{:10s} | {:6s} | {:7s}".format('Type', 'Total', 'FPS'))
        for ty, total in all_results[test_name].iteritems():
            print("{:10s} | {:6.2f} | {:7.2f}".format(
                ty, total, frame_count[test_name] / total))

    print(all_results)
    return all_results


def kernel_sol(tests):
    def run_kernel_trial(operation, path, frames):
        print('Running kernel trial: {}'.format(path))
        clear_filesystem_cache()
        current_env = os.environ.copy()
        start = time.time()
        program_path = os.path.join(
            COMPARISON_DIR, 'build/kernel_sol/kernel_sol')
        p = subprocess.Popen([
            program_path,
            '--operation', operation,
            '--frames', str(frames),
            '--path', path,
        ], env=current_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        so, se = p.communicate()
        rc = p.returncode
        elapsed = time.time() - start
        timings = {}
        if rc != 0:
            print('Trial FAILED after {:.3f}s'.format(elapsed))
            print(so)
            elapsed = -1
        else:
            print('Trial succeeded, took {:.3f}s'.format(elapsed))
            for line in so.splitlines():
                if line.startswith('TIMING: '):
                    k, s, v = line[len('TIMING: '):].partition(",")
                    timings[k] = float(v)
            elapsed = timings['total']
        return elapsed, timings

    operations = ['histogram', 'flow', 'caffe']
    iters = {'histogram': 50,
             'flow': 5,
             'caffe': 30}

    all_results = {}
    for test_name, paths in tests.iteritems():
        assert(len(paths) == 1)
        path = paths[0]

        all_results[test_name] = {}

        frames = 512

        for op in operations:
            all_results[test_name][op] = run_kernel_trial(
                op, path, frames)

        print('Kernel SOL on ', test_name)
        print("{:10s} | {:6s} | {:7s}".format('Kernel', 'Total', 'FPS'))
        for ty, (total, _) in all_results[test_name].iteritems():
            tot_frames = frames * iters[ty]
            print("{:10s} | {:6.2f} | {:7.2f}".format(
                ty, total, tot_frames / (1.0 * total)))

    print(all_results)
    return all_results


def effective_io_rate_benchmark():
    dataset_name = 'kcam_benchmark'
    in_job_name = scanner.Scanner.base_job_name()
    pipeline_name = 'effective_io_rate'
    out_job_name = 'eir_test'
    trial_settings = [{'force': True,
                       'node_count': 1,
                       'pus_per_node': 1,
                       'work_item_size': wis,
                       'load_workers_per_node': workers,
                       'save_workers_per_node': 1}
                      for wis in [64, 128, 256, 512, 1024, 2048, 4096, 8096]
                      for workers in [1, 2, 4, 8, 16]]
    results = []
    io = []
    for settings in trial_settings:
        result = run_trial(dataset_name, in_job_name, pipeline_name,
                           out_job_name, settings)
        io.append(get_trial_total_io_read(result) / (1024 * 1024)) # to mb
        results.append(result)
    rows = [{
        'work_item_size': y['work_item_size'],
        'load_workers_per_node': y['load_workers_per_node'],
        'time': x[0],
        'MB': i,
        'MB/s': i/x[0],
        'Effective MB/s': io[-1]/x[0]
    } for x, i, y in zip(results, io, trial_settings)]
    output_csv = dicts_to_csv(['work_item_size',
                               'load_workers_per_node',
                               'time',
                               'MB',
                               'MB/s',
                               'Effective MB/s'],
                              rows)
    print('Effective IO Rate Trials')
    print(output_csv)


def get_trial_total_decoded_frames(result):
    total_time, profilers = result

    total_decoded_frames = 0
    total_effective_frames = 0
    # Per node
    for node, (_, profiler) in profilers.iteritems():
        for prof in profiler['eval']:
            c = prof['counters']
            total_decoded_frames += (
                c['decoded_frames'] if 'decoded_frames' in c else 0)
            total_effective_frames += (
                c['effective_frames'] if 'effective_frames' in c else 0)
    return total_decoded_frames, total_effective_frames


def effective_decode_rate_benchmark():
    dataset_name = 'anewhope'
    in_job_name = scanner.Scanner.base_job_name()
    pipeline_name = 'effective_decode_rate'
    out_job_name = 'edr_test'
    trial_settings = [{'force': True,
                       'node_count': 1,
                       'pus_per_node': pus,
                       'work_item_size': wis,
                       'load_workers_per_node': 1,
                       'save_workers_per_node': 1}
                      for wis in [128, 256, 512, 1024, 2048, 4096]
                      for pus in [1, 2]]
    results = []
    decoded_frames = []
    for settings in trial_settings:
        result = run_trial(dataset_name, in_job_name, pipeline_name,
                           out_job_name, settings)
        decoded_frames.append(get_trial_total_decoded_frames(result))
        results.append(result)

    rows = [{
        'work_item_size': y['work_item_size'],
        'pus_per_node': y['pus_per_node'],
        'time': x[0],
        'decoded_frames': d[0],
        'effective_frames': d[1],
    } for x, d, y in zip(results, decoded_frames, trial_settings)]
    output_csv = dicts_to_csv(['work_item_size',
                               'pus_per_node',
                               'time',
                               'decoded_frames',
                               'effective_frames'],
                              rows)
    print('Effective Decode Rate Trials')
    print(output_csv)


def dnn_rate_benchmark():
    dataset_name = 'benchmark_kcam_dnn'
    pipeline_name = 'dnn_rate'
    out_job_name = 'dnnr_test'

    nets = [
        ('features/squeezenet.toml', [32, 64, 128]),
        ('features/alexnet.toml', [128, 256, 512]),
        ('features/googlenet.toml', [48, 96, 192]),
        ('features/resnet.toml', [8, 16, 32]),
        ('features/fcn8s.toml', [2, 4, 6]),
    ]
    trial_settings = [{'force': True,
                       'node_count': 1,
                       'pus_per_node': 1,
                       'work_item_size': max(batch_size * 2, 128),
                       'load_workers_per_node': 4,
                       'save_workers_per_node': 4,
                       'env': {
                           'SC_NET': net,
                           'SC_BATCH_SIZE': str(batch_size),
                       }}
                      for net, batch_sizes in nets
                      for batch_size in batch_sizes]
    results = []
    decoded_frames = []
    for settings in trial_settings:
        result = run_trial(dataset_name, in_job_name, pipeline_name,
                           out_job_name, settings)
        decoded_frames.append(get_trial_total_decoded_frames(result))
        results.append(result)

    rows = [{
        'net': y['env']['SC_NET'],
        'batch_size': y['env']['SC_BATCH_SIZE'],
        'time': x[0],
        'frames': d[1],
        'ms/frame': ((x[0] * 1000) / d[1]) if d[1] != 0 else 0
    } for x, d, y in zip(results, decoded_frames, trial_settings)]
    out_csv = dicts_to_csv(['net', 'batch_size', 'time', 'frames', 'ms/frame'],
                           rows)
    print('DNN Rate Trials')
    print(out_csv)


nets = [
    #['alex_net', 'features/alex_net.toml'],
    #['resnet', 'features/resnet.toml'],
    #['googlenet', 'features/googlenet.toml'],
    #['fcn', 'features/fcn8s.toml'],
    ['vgg', 'features/vgg.toml'],
]

def caffe_benchmark_cpu_trials():
    trial_settings = [
        {'net': nets[0][0],
         'net_descriptor_file': nets[0][1],
         'device_type': 'CPU',
         'net_input_width': -1,
         'net_input_height': -1,
         'num_elements': 256,
         'batch_size': batch_size}
        for batch_size in [1, 2, 4, 8, 16, 32]
    ] + [
        {'net': net[0],
         'net_descriptor_file': net[1],
         'device_type': 'CPU',
         'net_input_width': -1,
         'net_input_height': -1,
         'num_elements': 64,
         'batch_size': batch_size}
        for net in nets
        for batch_size in [1, 2, 4, 8, 16]]
    times = []
    for settings in trial_settings:
        trial_times = []
        for i in range(5):
            t = run_caffe_trial(**settings)
            trial_times.append(t)
        times.append(trial_times)

    print_caffe_trial_times('Caffe Throughput Benchmark', trial_settings, times)


def caffe_benchmark_gpu_trials():
    trial_settings = [
        {'net': nets[0][0],
         'net_descriptor_file': nets[0][1],
         'device_type': 'GPU',
         'net_input_width': -1,
         'net_input_height': -1,
         'num_elements': 4096,
         'batch_size': batch_size}
        for batch_size in [1, 2, 4, 8, 16, 32]
    ] + [
        {'net': net[0],
         'net_descriptor_file': net[1],
         'device_type': 'GPU',
         'net_input_width': -1,
         'net_input_height': -1,
         'num_elements': 2048,
         'batch_size': batch_size}
        for net in nets
        for batch_size in [1, 2, 4, 8, 16, 32]]
    times = []
    for settings in trial_settings:
        trial_times = []
        for i in range(5):
            t = run_caffe_trial(**settings)
            trial_times.append(t)
        times.append(trial_times)

    print_caffe_trial_times('Caffe Throughput Benchmark', trial_settings, times)


def single_node_comparison_benchmark():
    def make_gather_frames(total_frames):
        return []

    def make_video_interval(total_frames):
        # A range 100th of the video every 10th of the video
        return [
            (f - (total_frames / 100), f)
            for f in range(total_frames / 10, total_frames + 1,
                           total_frames / 10)]

    small_video = '/n/scanner/wcrichto.new/videos/movies/private/meanGirls.mp4'
    small_video_frames = 139301
    small_samplings = {
        'all': ('all',),
        'strided_short': ('strided', 30),
        'strided_long': ('strided', 500),
        'gather': ('gather', []),
        'range': ('range', make_video_interval(small_video_frames)),
        'hist_cpu_all': ('range', [[0, small_video_frames / 4]]),
        'caffe_all': ('range', [[0, small_video_frames / 4]]),
        'flow_cpu_all': ('range', [[0, small_video_frames / 100]]),
        'flow_gpu_all': ('range', [[0, small_video_frames / 20]]),
    }

    large_video = '/n/scanner/wcrichto.new/videos/movies/private/fightClub.mp4'
    large_video_frames = 200158
    large_samplings = {
        'all': ('all',),
        'strided_short': ('strided', 30),
        'strided_long': ('strided', 500),
        'gather': ('gather', []),
        'range': ('range', make_video_interval(large_video_frames)),
        'hist_cpu_all': ('range', [[0, large_video_frames / 4]]),
        'caffe_all': ('range', [[0, large_video_frames / 4]]),
        'flow_cpu_all': ('range', [[0, large_video_frames / 100]]),
        'flow_gpu_all': ('range', [[0, large_video_frames / 20]]),
    }

    tests = [
    {'name': 'flow_cpu',
     'sampling': 'flow_cpu_all',
     'scanner_settings': {
         'item_size': 64,
         'cpu_pool': None,
         'pipeline_instances_per_node': 32
     }},
        {'name': 'flow_gpu',
         'sampling': 'flow_gpu_all',
         'scanner_settings': {
             'item_size': 1024,
             'gpu_pool': '3G',
             'pipeline_instances_per_node': 1
         }},
        {'name': 'histogram_cpu',
         'sampling': 'hist_cpu_all',
         'scanner_settings': {
             'item_size': 1024,
             'cpu_pool': '32G',
             'pipeline_instances_per_node': 16
         }},
        {'name': 'histogram_gpu',
         'sampling': 'all',
         'scanner_settings': {
             'item_size': 2048,
             'gpu_pool': '6G',
             'pipeline_instances_per_node': 1
         }},
        {'name': 'caffe',
         'sampling': 'caffe_all',
         'scanner_settings': {
             'item_size': 2048,
             'gpu_pool': '4G',
             'pipeline_instances_per_node': 1
         }},
        # {'name': 'strided_hist_cpu',
        #  'sampling': 'strided_short',
        #  'scanner_settings': {
        #      'item_size': 2048,
        #      'gpu_pool': '6G',
        #      'pipeline_instances_per_node': 1
        #  }},
        # {'name': 'strided_hist_cpu',
        #  'ops': hist_cpu,
        #  'frame_factor': 1,
        #  'scanner_settings': {
        #      'item_size': 2048,
        #      'gpu_pool': '6G',
        #      'pipeline_instances_per_node': 1
        #  }},
    {'name': 'gather_hist_gpu',
     'sampling': 'strided_long',
     'scanner_settings': {
         'item_size': 4096,
         'gpu_pool': '4G',
         'pipeline_instances_per_node': 1
     }},
        {'name': 'gather_hist_cpu',
         'sampling': 'strided_long',
         'scanner_settings': {
             'item_size': 4,
             'cpu_pool': '90G',
             'pipeline_instances_per_node': 8
         }},
        # {'name': 'gather_hist_cpu',
        #  'ops': hist_gpu,
        #  'frame_factor': 1,
        #  'scanner_settings': {
        #      'item_size': 2048,
        #      'gpu_pool': '6G',
        #      'pipeline_instances_per_node': 1
        #  }},
        # {'name': 'gather_hist_gpu',
        #  'ops': hist_gpu,
        #  'frame_factor': 1,
        #  'scanner_settings': {
        #      'item_size': 2048,
        #      'gpu_pool': '6G',
        #      'pipeline_instances_per_node': 1
        #  }},
        # {'name': 'range_hist_cpu',
        #  'ops': hist_gpu,
        #  'frame_factor': 1,
        #  'scanner_settings': {
        #      'item_size': 2048
        #      'gpu_pool': '6G',
        #      'pipeline_instances_per_node': 1
        #  }},
    # {'name': 'range_hist_gpu',
    #  'sampling': 'range',
    #  'scanner_settings': {
    #      'item_size': 2048,
    #      'gpu_pool': '6G',
    #      'pipeline_instances_per_node': 1
    #  }}
    ]

    # tests = [
    # {'name': 'flow_cpu',
    #  'sampling': 'flow_cpu_all',
    #  'scanner_settings': {
    #      'item_size': 256,
    #      'cpu_pool': None,
    #      'pipeline_instances_per_node': 1
    #  }}]

    # tests = [
    #     {'name': 'gather_hist_gpu',
    #      'sampling': 'strided_long',
    #      'scanner_settings': {
    #          'item_size': 4096,
    #          'gpu_pool': '4G',
    #          'pipeline_instances_per_node': 1
    #      }},
    #     {'name': 'gather_hist_cpu',
    #      'sampling': 'strided_long',
    #      'scanner_settings': {
    #          'item_size': 4,
    #          'cpu_pool': '90G',
    #          'pipeline_instances_per_node': 8
    #      }},
    # ]

    stests = []
    for t in tests:
        x = t.copy()
        x['sampling'] = small_samplings[t['sampling']]
        stests.append(x)

    #scanner_results = scanner_benchmark(small_video, small_video_frames, stests)
    scanner_results = {'caffe': [{'frames': 139301, 'results': (124.215575869, {'eval': {'caffe:net': '92.931810', 'caffe:transform_input': '10.751346', 'decode': '123.109152', 'evaluate': '105.083320', 'idle': '169.292046', 'init': '0.002564', 'memcpy': '0.111664', 'op_marshal': '0.199384', 'setup': '3.441307', 'task': '105.757011'}, 'load': {'idle': '249.801238', 'io': '4.098090', 'setup': '0.000026', 'task': '5.091766'}, 'save': {'idle': '265.077130', 'io': '2.716596', 'setup': '0.000023', 'task': '2.718703'}, 'total_time': '124.215576'})}], 'flow_cpu': [{'frames': 696, 'results': (13.034195915, {'eval': {'decode': '45.335426', 'evaluate': '165.289857', 'idle': '780.737544', 'init': '1.530940', 'op_marshal': '0.000214', 'setup': '2.700304', 'task': '165.330727'}, 'load': {'idle': '32.027327', 'io': '0.020475', 'setup': '0.000058', 'task': '0.762892'}, 'save': {'idle': '45.890137', 'io': '0.034020', 'setup': '0.000113', 'task': '0.039795'}, 'total_time': '13.034196'})}], 'flow_gpu': [{'frames': 6965, 'results': (45.439282773, {'eval': {'decode': '43.106413', 'evaluate': '45.140732', 'idle': '74.221661', 'init': '0.000686', 'memcpy': '0.009722', 'op_marshal': '0.014059', 'setup': '1.612254', 'task': '45.213637'}, 'load': {'idle': '53.903756', 'io': '0.244350', 'setup': '0.000059', 'task': '0.341566'}, 'save': {'idle': '105.342207', 'io': '0.288177', 'setup': '0.000010', 'task': '0.288527'}, 'total_time': '45.439283'})}], 'histogram_cpu': [{'frames': 139301, 'results': (32.297165602, {'eval': {'decode': '483.722451', 'evaluate': '408.317141', 'idle': '1074.992327', 'init': '2.765437', 'op_marshal': '0.002170', 'setup': '5.705983', 'task': '412.049338'}, 'load': {'idle': '20.057779', 'io': '5.851161', 'setup': '0.000032', 'task': '8.196627'}, 'save': {'idle': '83.579506', 'io': '0.861688', 'setup': '0.000069', 'task': '0.875858'}, 'total_time': '32.297166'})}], 'histogram_gpu': [{'frames': 139301, 'results': (33.553154557, {'eval': {'decode': '32.609033', 'evaluate': '14.324457', 'idle': '82.179450', 'init': '0.003078', 'memcpy': '0.106341', 'op_marshal': '0.201628', 'setup': '0.758748', 'task': '14.852671'}, 'load': {'idle': '78.701292', 'io': '3.914725', 'setup': '0.000020', 'task': '4.976457'}, 'save': {'idle': '85.108238', 'io': '1.875328', 'setup': '0.000112', 'task': '1.877589'}, 'total_time': '33.553155'})}], 'range_hist_gpu': [{'frames': 13930, 'results': (3.597061053, {'eval': {'decode': '3.302835', 'evaluate': '1.558980', 'idle': '35.531154', 'init': '0.000488', 'memcpy': '0.013573', 'op_marshal': '0.023253', 'setup': '0.149820', 'task': '1.613436'}, 'load': {'idle': '23.547049', 'io': '0.479616', 'setup': '0.000018', 'task': '0.612952'}, 'save': {'idle': '26.583728', 'io': '0.198355', 'setup': '0.000023', 'task': '0.198698'}, 'total_time': '3.597061'})}], 'strided_hist_long_gpu': [{'frames': 464, 'results': (3.716887399, {'eval': {'decode': '2.495283', 'evaluate': '0.056301', 'idle': '38.109545', 'init': '0.000893', 'memcpy': '0.003860', 'op_marshal': '0.004425', 'setup': '0.192161', 'task': '0.062307'}, 'load': {'idle': '10.001863', 'io': '0.824274', 'setup': '0.000027', 'task': '0.926700'}, 'save': {'idle': '13.622593', 'io': '0.025920', 'setup': '0.000119', 'task': '0.026030'}, 'total_time': '3.716887'})}], 'strided_hist_short_gpu': [{'frames': 4643, 'results': (28.891601876, {'eval': {'decode': '24.867581', 'evaluate': '0.497028', 'idle': '89.946550', 'init': '0.000483', 'memcpy': '0.007095', 'op_marshal': '0.017515', 'setup': '1.184723', 'task': '0.532347'}, 'load': {'idle': '20.007012', 'io': '5.262028', 'setup': '0.000045', 'task': '6.076579'}, 'save': {'idle': '74.641240', 'io': '0.068315', 'setup': '0.000005', 'task': '0.068445'}, 'total_time': '28.891602'})}]}
    standalone_results = {k: [{'frames': v[0]['frames'], 'results': (-1, {})}]
                          for k, v in scanner_results.iteritems()}
    peak_results = scanner_results
    # graph.comparison_graphs('small', 640, 480, standalone_results, scanner_results,
    #                         peak_results)

    ltests = []
    for t in tests:
        x = t.copy()
        x['sampling'] = large_samplings[t['sampling']]
        ltests.append(x)

    #standalone_results = standalone_benchmark(large_video, large_video_frames, ltests)
    standalone_results = {'histogram_gpu': [{'frames': 200158, 'results': (208.96, {'load': 154.65, 'total': 208.96, 'setup': 0.15, 'save': 0.37, 'eval': 53.59})}], 'gather_hist_gpu': [{'frames': 400, 'results': (209.62, {'load': 0.33, 'total': 209.62, 'setup': 0.15, 'save': 0.0, 'eval': 0.12})}], 'flow_cpu': [{'frames': 333, 'results': (140.32, {'load': 1.41, 'total': 140.32, 'setup': 0.13, 'save': 0.0, 'eval': 138.91})}], 'histogram_cpu': [{'frames': 50039, 'results': (367.73, {'load': 145.82, 'total': 367.73, 'setup': 0.13, 'save': 0.05, 'eval': 221.79})}], 'flow_gpu': [{'frames': 10007, 'results': (232.46, {'load': 0.77, 'total': 232.46, 'setup': 0.28, 'save': 0.0, 'eval': 231.51})}], 'gather_hist_cpu': [{'frames': 400, 'results': (51.35, {'load': 2.37, 'total': 51.35, 'setup': 0.13, 'save': 0.0, 'eval': 3.73})}], 'caffe': [{'frames': 200158, 'results': (-1, {})}]}
    scanner_results = {'caffe': [{'frames': 50039, 'results': (126.863404986, {'eval': {'caffe:net': '33.357140', 'caffe:transform_input': '66.319959', 'decode': '125.769289', 'evaluate': '100.174348', 'frames_decoded': 51320, 'frames_fed': 51643, 'frames_used': 50039, 'idle': '176.178832', 'init': '0.000490', 'memcpy': '0.020457', 'op_marshal': '0.030317', 'setup': '7.520566', 'task': '100.236231'}, 'load': {'idle': '229.027299', 'io': '2.452382', 'io_read': 507321797, 'setup': '0.001674', 'task': '3.056205'}, 'save': {'idle': '270.053932', 'io': '0.913868', 'io_write': 200556312, 'setup': '0.000248', 'task': '0.914203'}, 'total_time': '126.863405'})}], 'flow_cpu': [{'frames': 2001, 'results': (63.839779358, {'eval': {'decode': '205.726926', 'evaluate': '1768.963717', 'frames_decoded': 4085, 'frames_fed': 3685, 'frames_used': 2001, 'idle': '3163.768038', 'init': '0.681177', 'op_marshal': '0.000039', 'setup': '3.821897', 'task': '1769.154424'}, 'load': {'idle': '20.040472', 'io': '0.133431', 'io_read': 57562914, 'setup': '0.000163', 'task': '0.563751'}, 'save': {'idle': '147.624472', 'io': '0.006925', 'io_write': 18009, 'setup': '0.000857', 'task': '0.007475'}, 'total_time': '63.839779'})}], 'flow_gpu': [{'frames': 10007, 'results': (248.626166466, {'eval': {'decode': '238.256139', 'evaluate': '248.121545', 'frames_decoded': 10342, 'frames_fed': 10482, 'frames_used': 10007, 'idle': '275.726441', 'init': '0.002060', 'memcpy': '0.009847', 'op_marshal': '0.012221', 'setup': '3.580473', 'task': '248.186052'}, 'load': {'idle': '296.323660', 'io': '0.539573', 'io_read': 112728829, 'setup': '0.000282', 'task': '0.678909'}, 'save': {'idle': '496.894921', 'io': '0.569726', 'io_write': 90063, 'setup': '0.000303', 'task': '0.569924'}, 'total_time': '248.626166'})}], 'gather_hist_cpu': [{'frames': 400, 'results': (21.549662711, {'eval': {'decode': '165.283746', 'evaluate': '3.290798', 'frames_decoded': 37481, 'frames_fed': 40082, 'frames_used': 401, 'idle': '573.573771', 'init': '0.043311', 'op_marshal': '0.000037', 'setup': '1.580372', 'task': '3.292639'}, 'load': {'idle': '20.008724', 'io': '2.915937', 'io_read': 412212668, 'setup': '0.000196', 'task': '3.555391'}, 'save': {'idle': '62.876299', 'io': '0.018731', 'io_write': 80200, 'setup': '0.000054', 'task': '0.020275'}, 'total_time': '21.549663'})}], 'gather_hist_gpu': [{'frames': 400, 'results': (50.547078065, {'eval': {'decode': '46.507375', 'evaluate': '0.098743', 'frames_decoded': 42586, 'frames_fed': 45454, 'frames_used': 401, 'idle': '132.330747', 'init': '0.000372', 'memcpy': '0.005163', 'op_marshal': '0.005359', 'setup': '2.500418', 'task': '0.104367'}, 'load': {'idle': '10.003470', 'io': '2.651054', 'io_read': 412212668, 'setup': '0.000407', 'task': '3.141374'}, 'save': {'idle': '60.482490', 'io': '0.015511', 'io_write': 80200, 'setup': '0.000209', 'task': '0.015551'}, 'total_time': '50.547078'})}], 'histogram_cpu': [{'frames': 50039, 'results': (51.12213506, {'eval': {'decode': '708.187778', 'evaluate': '698.622058', 'frames_decoded': 52611, 'frames_fed': 52814, 'frames_used': 50039, 'idle': '1315.758985', 'init': '0.255978', 'op_marshal': '0.000204', 'setup': '2.810293', 'task': '698.773590'}, 'load': {'idle': '20.046779', 'io': '3.222126', 'io_read': 533254426, 'setup': '0.000052', 'task': '5.084221'}, 'save': {'idle': '119.417376', 'io': '0.067553', 'io_write': 10007800, 'setup': '0.000030', 'task': '0.068376'}, 'total_time': '51.122135'})}], 'histogram_gpu': [{'frames': 200158, 'results': (215.634761139, {'eval': {'decode': '213.731645', 'evaluate': '39.514628', 'frames_decoded': 205543, 'frames_fed': 206534, 'frames_used': 200158, 'idle': '418.698448', 'init': '0.000948', 'memcpy': '0.086007', 'op_marshal': '0.135303', 'setup': '4.463794', 'task': '39.757738'}, 'load': {'idle': '418.835705', 'io': '9.484100', 'io_read': 1972359963, 'setup': '0.000232', 'task': '11.587169'}, 'save': {'idle': '446.926429', 'io': '2.214957', 'io_write': 40031600, 'setup': '0.000180', 'task': '2.216198'}, 'total_time': '215.634761'})}]}
    #scanner_results = scanner_benchmark(large_video, large_video_frames, ltests)
    # standalone_results = {k: [{'frames': v[0]['frames'], 'results': (-1, {})}]
    #                       for k, v in scanner_results.iteritems()}
    # standalone_results['histogram_cpu'][0]['results'] = (1516.348484848, {})
    # standalone_results['histogram_gpu'][0]['results'] = (209.809224319, {})
    # standalone_results['flow_cpu'][0]['results'] = (716.195372751, {})
    # standalone_results['flow_gpu'][0]['results'] = (162.672109, {})
    standalone_results['caffe'][0]['results'] = (1021.214285714, {})
    peak_results = copy.deepcopy(scanner_results)
    peak_results['histogram_cpu'][0]['results'] = (50.692014531, {})
    peak_results['histogram_gpu'][0]['results'] = (209.73 , {})
    # peak_results['flow_cpu'][0]['results'] = (53.517230781, {})
    peak_results['flow_gpu'][0]['results'] = (233.742059043, {})
    peak_results['caffe'][0]['results'] = (116.913551402, {})
    graph.comparison_graphs('large', 1920, 800, standalone_results, scanner_results,
                            peak_results)


def striding_comparison_benchmark():
    # oliver: 157, 0.55
    # kcam: 29, 0.0
    # small: 23.1, 0.05
    # large: 74.29, 0.33
    videos = [
         #('oliver', '/n/scanner/apoms/videos/oliver_trump_720p.mp4',
         # 52176),
        #('kcam', '/n/scanner/wcrichto.new/videos/kcam/20150308_205310_836.mp4',
          # 52542),
        #('small', '/n/scanner/wcrichto.new/videos/movies/private/meanGirls.mp4',
        # 139301),
        ('large', '/n/scanner/wcrichto.new/videos/movies/private/fightClub.mp4',
          200158),
    ]

    strides = [1, 5, 15] + range(30, 391, 60)
    tests = [{'name': '{:d}'.format(stride),
              'sampling': ('strided', stride),
              'scanner_settings': {
                  'item_size': 10000,
                  'gpu_pool': '4G',
                  'pipeline_instances_per_node': 1
              }}
             for stride in strides]

    results = []
    for name, video, frames in videos:
        r = scanner_striding_test(video, frames, tests)
        results.append((name, r))
    pprint(results)
    graph.striding_comparison_graphs(strides, results)


def multi_gpu_comparison_benchmark():
    small_video = '/n/scanner/wcrichto.new/videos/movies/private/meanGirls.mp4'
    small_video_frames = 139301
    large_video = '/n/scanner/wcrichto.new/videos/movies/private/fightClub.mp4'
    large_video_frames = 200158

    videos = [
        ('small', small_video, small_video_frames),
        ('large', large_video, large_video_frames),
    ]

    # base_tests = [
    #     {'name': 'flow_gpu',
    #      'sampling': ('all',),
    #      'scanner_settings': {
    #          'item_size': 1024,
    #          'gpu_pool': '3G',
    #      }},
    #     {'name': 'histogram_gpu',
    #      'sampling': ('all',),
    #      'scanner_settings': {
    #          'item_size': 2048,
    #          'gpu_pool': '6G',
    #      }},
    #     {'name': 'caffe',
    #      'sampling': ('all',),
    #      'scanner_settings': {
    #          'item_size': 2048,
    #          'gpu_pool': '4G',
    #      }}
    # ]


    base_tests = [
        {'name': 'flow_gpu',
         'sampling': ('range', [[0, 10000]]),
         'scanner_settings': {
             'item_size': 1024,
             'gpu_pool': '3G',
         }},
        {'name': 'histogram_gpu',
         'sampling': ('range', [[0, 100000]]),
         'scanner_settings': {
             'item_size': 2048,
             'gpu_pool': '6G',
         }},
        {'name': 'caffe',
         'sampling': ('range', [[0, 100000]]),
         'scanner_settings': {
             'item_size': 2048,
             'gpu_pool': '5G',
         }}
    ]

    gpus = [1, 2, 4]
    tests = []
    for g in gpus:
        for t in base_tests:
            b = copy.deepcopy(t)
            b['name'] += "_{:d}".format(g)
            b['scanner_settings']['pipeline_instances_per_node'] = 1
            b['scanner_settings']['nodes'] = ['localhost:500{:d}'.format(i + 2)
                                              for i in range(g)]
            tests.append(b)

    for name, video, frames in videos:
         r = scanner_benchmark(video, frames, tests)
         pprint(name)
         pprint(r)
         graph.multi_gpu_comparison_graphs(name, gpus, r)


def micro_comparison_driver():

    #decode_sol(tests, frame_counts)
    tests = {
        #'fight': ['/n/scanner/wcrichto.new/videos/movies/private/fightClub.mp4'],
        #'excalibur': ['/n/scanner/wrichto.new/videos/movies/excalibur.mp4'],
        'mean': ['/n/scanner/wcrichto.new/videos/movies/private/meanGirls.mp4'],
    }
    frame_counts = {'charade': 163430,
                    'fight': 200158,
                    'excalibur': 202275,
                    'mean': 139301
    }
    if 0:
        #decode_sol(tests, frame_counts)
        kernel_sol(tests)


BENCHMARKS = {
    'multi_gpu': multi_gpu_benchmark,
    'standalone': standalone_benchmark,
    'scanner': scanner_benchmark,
    'peak': peak_benchmark,
}


def bench_main(args):
    single_node_comparison_benchmark()
    #striding_comparison_benchmark()
    #multi_gpu_comparison_benchmark()
    exit()
    test = args.test
    out_dir = args.output_directory
    if test == 'all':
        for name, fn in BENCHMARKS.iteritems():
            fn()
    else:
        fn = BENCHMARKS[test]
        fn()
