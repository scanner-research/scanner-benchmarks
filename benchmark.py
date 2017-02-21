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
from scannerpy import Database, DeviceType, Config
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DEVNULL = open(os.devnull, 'wb', 0)


def clear_filesystem_cache():
    os.system('sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"')


def run_trial(tasks, pipeline, collection_name, opts={}):
    print('Running trial: collection {:s} ...:'.format(collection_name))
    # Clear cache
    config_path = opts['config_path'] if 'config_path' in opts else None
    db_path = opts['db_path'] if 'db_path' in opts else None
    config = Config(config_path)
    if db_path is not None:
        config.db_path = db_path
    db = scanner.Database(config=config)
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
        out_collection = db.run(tasks, pipeline, collection_name,
                                **scanner_opts)
        prof = out_collection.profiler()
        elapsed = now() - start
        total = prof.total_time_interval()
        t = (total[1] - total[0])
        t /= float(1e9)  # ns to s
        print('Trial succeeded: {:.3f}s elapsed, {:.3f}s effective'.format(
            elapsed, t))
    except ScannerException:
        elapsed = now() - start
        success = False
        prof = None
        print('Trial FAILED after {:.3f}s'.format(elapsed))
        t = -1
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
        SCRIPT_DIR, 'build/debug/comparison/opencv/opencv_compare')
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

    db = scanner.Scanner()
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


def standalone_benchmark(tests, frame_counts, wh):
    output_dir = '/tmp/standalone'
    test_output_dir = '/tmp/outputs'
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

    def run_standalone_trial(input_type, paths_file, operation):
        print('Running standalone trial: {}, {}, {}'.format(
            input_type,
            paths_file,
            operation))
        clear_filesystem_cache()
        current_env = os.environ.copy()
        start = time.time()
        program_path = os.path.join(
            SCRIPT_DIR, '../build/comparison/standalone/standalone_comparison')
        p = subprocess.Popen([
            program_path,
            '--input_type', input_type,
            '--paths_file', paths_file,
            '--operation', operation
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

    operations = ['histogram_cpu', 'histogram_gpu', 'flow_cpu', 'flow_gpu',
                  'caffe']

    bmp_template = "ffmpeg -i {input} -start_number 0 {output}/frame%07d.bmp"
    jpg_template = "ffmpeg -i {input} -start_number 0 {output}/frame%07d.jpg"

    all_results = {}
    for test_name, paths in tests.iteritems():
        all_results[test_name] = {}
        for op in operations:
            all_results[test_name][op] = []

        # # bmp
        # os.system('rm -rf {}'.format(output_dir))
        # os.system('mkdir {}'.format(output_dir))
        # run_paths = []
        # for p in paths:
        #     base = os.path.basename(p)
        #     run_path = os.path.join(output_dir, base)
        #     os.system('mkdir -p {}'.format(run_path))
        #     run_cmd(bmp_template, {
        #         'input': p,
        #         'output': run_path
        #     })
        #     meta = read_meta(run_path)
        #     write_meta_file(run_path, meta)
        #     run_paths.append(run_path)
        # write_paths(run_paths)

        # for op in operations:
        #     all_results[test_name][op].append(
        #         run_standalone_trial('bmp', paths_file, op))

        # # # jpg
        # os.system('rm -rf {}'.format(output_dir))
        # os.system('mkdir {}'.format(output_dir))
        # run_paths = []
        # for p in paths:
        #     base = os.path.basename(p)
        #     run_path = os.path.join(output_dir, base)
        #     os.system('mkdir -p {}'.format(run_path))
        #     run_cmd(jpg_template, {
        #         'input': p,
        #         'output': run_path
        #     })
        #     meta = read_meta(run_path)
        #     write_meta_file(run_path, meta)
        #     run_paths.append(run_path)
        # write_paths(run_paths)

        # for op in operations:
        #     all_results[test_name][op].append(
        #         run_standalone_trial('jpg', paths_file, op))

        # video
        for op in operations:
            os.system('rm -rf {}'.format(output_dir))
            os.system('mkdir {}'.format(output_dir))

            run_paths = []
            for p in paths:
                base = os.path.basename(p)
                run_path = os.path.join(output_dir, base)
                os.system('cp {} {}'.format(p, run_path))
                run_paths.append(run_path)
            write_paths(run_paths)

            frames = frame_counts[test_name]
            if op == 'flow_cpu':
                frames /= 200
            if op == 'flow_gpu':
                frames /= 20

            os.system('rm -rf {}'.format(test_output_dir))
            os.system('mkdir -p {}'.format(test_output_dir))
            all_results[test_name][op].append(
                {'results':run_standalone_trial('mp4', paths_file, op),
                 'frames': frames})

    print(all_results)
    return all_results


def scanner_benchmark(tests, frame_counts, wh):
    db_dir = '/tmp/scanner_db'

    db = scanner.Scanner()
    db._db_path = db_dir
    scanner_settings = {
        'db_path': db_dir,
        'node_count': 1,
        'pus_per_node': 1,
        'work_item_size': 64,
        'tasks_in_queue_per_pu': 3,
        'force': True,
        'env': {}
    }
    dataset_name = 'test'
    raw_job = 'raw_job'
    jpg_job = 'jpg_job'
    video_job = 'base'

    # Histogram benchmarks
    hist_cpu = db.ops.Histogram(device=DeviceType.CPU)
    hist_gpu = db.ops.Histogram(device=DeviceType.GPU)

    # Optical flow benchmarks
    of_cpu = db.ops.OpticalFlow(device=DeviceType.CPU)
    of_gpu = db.ops.Histogram(device=DeviceType.GPU)

    # Caffe benchmark
    descriptor = NetDescriptor.from_file(db, 'features/googlenet.toml')
    caffe_args = facenet_args.caffe_args
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

    caffe_cpu = caffe_pipeline(device=DeviceType.CPU)
    caffe_gpu = caffe_pipeline(device=DeviceType.GPU)

    operations = [('flow_gpu', of_cpu),
                  ('flow_cpu', of_gpu),
                  ('histogram_cpu', hist_cpu),
                  ('histogram_gpu', hist_gpu),
                  ('caffe', caffe_gpu)]

    all_results = {}
    for test_name, paths in tests.iteritems():
        all_results[test_name] = {}
        for op, _, _  in operations:
            all_results[test_name][op] = []

        os.system('rm -rf {}'.format(db_dir))
        # ingest data
        result, _ = db.ingest('video', dataset_name, paths, scanner_settings)
        assert(result)

        stride = 30
        scanner_settings['env']['SC_JOB_NAME'] = video_job
        for op, pipeline, dev in operations:
            frames = frame_counts[test_name]

            scanner_settings['env']['SC_DEVICE'] = dev
            scanner_settings['use_pool'] = True
            scanner_settings['pus_per_node'] = 1
            if op == 'histogram_cpu':
                scanner_settings['use_pool'] = False
                scanner_settings['tasks_in_queue_per_pu'] = 2
                if wh[test_name]['width'] == 640:
                    scanner_settings['io_item_size'] = 2048
                    scanner_settings['work_item_size'] = 1024
                    scanner_settings['pus_per_node'] = 4
                else:
                    scanner_settings['io_item_size'] = 1024
                    scanner_settings['work_item_size'] = 128
                    scanner_settings['pus_per_node'] = 8
            elif op == 'histogram_cpu_strided':
                scanner_settings['env']['SC_STRIDE'] = str(stride)
                scanner_settings['use_pool'] = False
                scanner_settings['tasks_in_queue_per_pu'] = 2
                if wh[test_name]['width'] == 640:
                    scanner_settings['io_item_size'] = 2048
                    scanner_settings['work_item_size'] = 1024
                    scanner_settings['pus_per_node'] = 4
                else:
                    scanner_settings['io_item_size'] = 1024
                    scanner_settings['work_item_size'] = 128
                    scanner_settings['pus_per_node'] = 4
            elif op == 'histogram_gpu':
                if wh[test_name]['width'] == 640:
                    scanner_settings['io_item_size'] = 2048
                    scanner_settings['work_item_size'] = 1024
                else:
                    scanner_settings['io_item_size'] = 512
                    scanner_settings['work_item_size'] = 128
            elif op == 'histogram_gpu_strided':
                scanner_settings['env']['SC_STRIDE'] = str(stride)
                if wh[test_name]['width'] == 640:
                    scanner_settings['io_item_size'] = 2048
                    scanner_settings['work_item_size'] = 1024
                else:
                    scanner_settings['io_item_size'] = 512
                    scanner_settings['work_item_size'] = 128
            elif op == 'flow_cpu':
                scanner_settings['use_pool'] = False
                frames /= 20
                scanner_settings['io_item_size'] = 16
                scanner_settings['work_item_size'] = 16
                scanner_settings['pus_per_node'] = 16
            elif op == 'flow_gpu':
                frames /= 20
                scanner_settings['io_item_size'] = 256
                # 96 hangs!
                #scanner_settings['io_item_size'] = 96
                scanner_settings['work_item_size'] = 64
            elif op == 'caffe':
                scanner_settings['io_item_size'] = 256
                scanner_settings['work_item_size'] = 64
            total, prof = run_trial(dataset_name, pipeline, op,
                                    scanner_settings)
            stats = generate_statistics(prof)
            all_results[test_name][op].append(
                {'results': (total, stats),
                 'frames': frames})

    print(all_results)
    return all_results


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
            SCRIPT_DIR, '../build/comparison/peak/peak_comparison')
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
            SCRIPT_DIR, '../build/comparison/ocv_decode/ocv_decode')
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
            SCRIPT_DIR, '../build/comparison/kernel_sol/kernel_sol')
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


def micro_comparison_driver():
    #pose_reconstruction_graphs({})
    tests = {
        'kcam': ['/n/scanner/wcrichto.new/videos/kcam/20150308_205310_836.mp4'],
        #'fight': ['/n/scanner/wcrichto.new/videos/movies/private/fightClub.mp4'],
        #'excalibur': ['/n/scanner/wrichto.new/videos/movies/excalibur.mp4'],
        #'mean': ['/n/scanner/wcrichto.new/videos/movies/private/meanGirls.mp4'],
    }
    frame_counts = {'charade': 163430,
                    'fight': 200158,
                    'excalibur': 202275,
                    'mean': 139301,
                    'kcam': 52542,
    }
    frame_wh = {'charade': {'width': 1920, 'height': 1080},
                'fight': {'width': 1920, 'height': 800},
                'excalibur': {'width': 1920, 'height': 1080},
                'mean': {'width': 640, 'height': 480},
                'kcam': {'width': 1280, 'height': 720},
    }
    t = 'fight'
    if 0:
        #standalone_caffe_compute(tests)
        peak_caffe_compute(tests, frame_counts, frame_wh)
        exit()



    #t = 'mean'
    t = 'fight'
    #peak_results = peak_benchmark(tests, frame_counts, frame_wh)
    #scanner_results = scanner_benchmark(tests, frame_wh)
    #peak_results = peak_benchmark(tests, frame_counts, frame_wh)
    if 1:
        #standalone_results = standalone_benchmark(tests, frame_counts, frame_wh)
        scanner_results = scanner_benchmark(tests, frame_counts, frame_wh)
        peak_results = peak_benchmark(tests, frame_counts, frame_wh)
        print(scanner_results)
        print(peak_results)
        #comparison_graphs(t, frame_counts, frame_wh, standalone_results,
        #                  scanner_results, peak_results)
    if 0:
        #640
        t = 'mean'
        standalone_results = {'mean': {'caffe': [(128.86, {'load': 34.5, 'save': 0.19, 'transform': 56.25, 'eval': 94.17, 'net': 37.91, 'total': 128.86})], 'flow': [(42.53, {'load': 0.13, 'total': 42.53, 'setup': 0.21, 'save': 5.44, 'eval': 17.58})], 'histogram': [(13.54, {'load': 7.05, 'total': 13.54, 'setup': 0.12, 'save': 0.05, 'eval': 6.32})]}}
        scanner_results = {'mean': {'caffe': [(44.495288089, {'load': {'setup': '0.000009', 'task': '2.174472', 'idle': '173.748807', 'io': '2.138090'}, 'save': {'setup': '0.000008', 'task': '1.072224', 'idle': '117.011795', 'io': '1.065889'}, 'eval': {'task': '84.702374', 'evaluate': '83.057708', 'setup': '4.623244', 'evaluator_marshal': '1.427507', 'decode': '42.458139', 'idle': '146.444473', 'caffe:net': '34.756799', 'caffe:transform_input': '5.282478', 'memcpy': '1.353623'}})], 'flow': [(34.700563742, {'load': {'setup': '0.000010', 'task': '0.654652', 'idle': '83.952595', 'io': '0.641715'}, 'save': {'setup': '0.000008', 'task': '6.257866', 'idle': '62.448266', 'io': '6.257244'}, 'eval': {'task': '20.600713', 'evaluate': '20.410105', 'setup': '2.016671', 'evaluator_marshal': '0.094336', 'decode': '1.044027', 'idle': '105.924678', 'memcpy': '0.089637', 'flowcalc': '17.262241'}})], 'histogram': [(15.449293653, {'load': {'setup': '0.000007', 'task': '2.212311', 'idle': '83.767484', 'io': '2.192515'}, 'save': {'setup': '0.000008', 'task': '0.659185', 'idle': '59.795770', 'io': '0.658409'}, 'eval': {'task': '20.127624', 'evaluate': '19.132718', 'setup': '2.043204', 'histogram': '4.870099', 'decode': '14.261789', 'idle': '97.814596', 'evaluator_marshal': '0.845618', 'memcpy': '0.827516'}})]}}
        peak_results = {'mean': {'caffe': [(41.27, {'feed': 40.9, 'load': 0.0, 'total': 41.27, 'transform': 3.28, 'decode': 40.97, 'idle': 9.04, 'eval': 37.21, 'net': 33.92, 'save': 0.34})], 'flow': [(29.91, {'feed': 15.62, 'load': 0.0, 'total': 29.91, 'decode': 0.85, 'eval': 18.21, 'save': 7.74})], 'histogram': [(12.3, {'feed': 12.25, 'load': 0.0, 'total': 12.3, 'setup': 0.0, 'decode': 12.26, 'eval': 4.11, 'save': 0.05})]}}
        comparison_graphs(t, frame_counts, frame_wh, standalone_results,
                          scanner_results, peak_results)
    if 1:
        #1920
        t = 'fight'
        #standalone_results = {'fight': {'caffe': [(252.92, {'load': 159.73, 'save': 0.19, 'transform': 55.09, 'eval': 93.0, 'net': 37.9, 'total': 252.92})], 'flow': [(57.03, {'load': 0.19, 'total': 57.03, 'setup': 0.23, 'save': 0.0, 'eval': 56.66})], 'histogram': [(52.41, {'load': 38.72, 'total': 52.41, 'setup': 0.16, 'save': 0.09, 'eval': 13.43})]}}
        #scanner_results = {'fight': {'caffe': [(134.643764659, {'load': {'setup': '0.000313', 'task': '2.732178', 'idle': '483.639105', 'io': '2.694216'}, 'save': {'setup': '0.000012', 'task': '0.998610', 'idle': '327.472855', 'io': '0.992970'}, 'eval': {'task': '227.921991', 'evaluate': '193.274604', 'setup': '6.592625', 'evaluator_marshal': '34.380673', 'decode': '99.733381', 'idle': '421.211272', 'caffe:net': '34.503387', 'caffe:transform_input': '58.372106', 'memcpy': '34.279553'}})], 'flow': [(68.246963461, {'load': {'setup': '0.000659', 'task': '0.232119', 'idle': '258.826515', 'io': '0.198524'}, 'save': {'setup': '0.000007', 'task': '0.003803', 'idle': '194.370179', 'io': '0.003245'}, 'eval': {'task': '72.947458', 'evaluate': '68.481038', 'setup': '2.086686', 'evaluator_marshal': '4.323619', 'decode': '4.343175', 'idle': '300.531346', 'memcpy': '4.314552', 'flowcalc': '61.589261'}})], 'histogram': [(61.569025441, {'load': {'setup': '0.000008', 'task': '2.827806', 'idle': '266.692132', 'io': '2.804325'}, 'save': {'setup': '0.000006', 'task': '0.615977', 'idle': '182.132882', 'io': '0.613140'}, 'eval': {'task': '68.860379', 'evaluate': '67.389445', 'setup': '2.053393', 'histogram': '7.404950', 'decode': '59.979797', 'idle': '293.526000', 'evaluator_marshal': '1.278468', 'memcpy': '1.234116'}})]}}
        #peak_results = {'fight': {'caffe': [(117.62, {'feed': 117.3, 'load': 0.0, 'total': 117.62, 'transform': 63.25, 'decode': 117.36, 'idle': 23.6, 'eval': 97.88, 'net': 34.63, 'save': 0.48})], 'flow': [(63.29, {'feed': 53.95, 'load': 0.0, 'total': 63.29, 'decode': 3.37, 'eval': 63.06, 'save': 2.44})], 'histogram': [(52.09, {'feed': 52.06, 'load': 0.0, 'total': 52.09, 'setup': 0.0, 'decode': 52.07, 'eval': 6.43, 'save': 0.53})]}}
        #standalone_results = {'fight': {'caffe': [(252.92, 50000, {})],
                                        # 'flow_gpu': [(57.03, 2500, {})],
                                        # 'flow_cpu': [(15.03, 250, {})],
                                        # 'histogram_cpu': [(15.41, 50000, {})],
                                        # 'histogram_gpu': [(52.41, 50000, {})]}}

        standalone_results = {'fight': {'histogram_gpu': [{'frames': 200158, 'results': (209.73, {'load': 155.6, 'total': 209.73, 'setup': 0.16, 'save': 0.37, 'eval': 53.52})}], 'caffe': [{'frames': 200158, 'results': (1016.33, {'load': 642.65, 'save': 0.62, 'transform': 222.14, 'eval': 373.05, 'net': 150.9, 'total': 1016.33})}], 'flow_cpu': [{'frames': 10007, 'results': (517.63, {'load': 4.06, 'total': 517.63, 'setup': 0.12, 'save': 0.0, 'eval': 513.56})}], 'histogram_cpu': [{'frames': 200158, 'results': (1508.85, {'load': 578.55, 'total': 1508.85, 'setup': 0.12, 'save': 0.2, 'eval': 929.99})}], 'flow_gpu': [{'frames': 10007, 'results': (233.72, {'load': 0.77, 'total': 233.72, 'setup': 0.25, 'save': 0.0, 'eval': 232.81})}]}}

        #scanner_results = {'fight': {'caffe': [(134.643764659, 50000, {})],
                                     # 'flow_gpu': [(68.246963461, 2500, {})],
                                     # 'flow_cpu': [(10.569025441, 250, {})],
                                     # 'histogram_cpu': [(40.569025441, 50000, {})],
                                     # 'histogram_gpu': [(52.569025441, 50000, {})]}}
        #peak_results = {'fight': {'caffe': [(120.643764659, 50000, {})],
                                  # 'flow_gpu': [(58.246963461, 2500, {})],
                                  # 'flow_cpu': [(8.569025441, 250, {})],
                                  # 'histogram_cpu': [(40.569025441, 50000, {})],
                                  # 'histogram_gpu': [(50.569025441, 50000, {})]}}
                           
        comparison_graphs(t, frame_counts, frame_wh, standalone_results,
                          scanner_results, peak_results)
    exit(1)

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


    tests = {
        #'fight': ['/n/scanner/wcrichto.new/videos/movies/fightClub.mp4'],
        # 'fight': [
        #     '/n/scanner/wcrichto.new/videos/movies/private/fightClub.mp4'
        #     #'/n/scanner/wcrichto.new/videos/movies/private/fightClub.mp4'
        # ],
        #'excalibur': ['/n/scanner/wrichto.new/videos/movies/excalibur.mp4'],
        'mean': [
            '/n/scanner/wcrichto.new/videos/movies/private/meanGirls.mp4',
            '/n/scanner/wcrichto.new/videos/movies/private/meanGirls.mp4'
        ],
    }
    frame_counts = {'charade': 163430,
                    'fight': 200158 * 1,
                    'excalibur': 202275,
                    'mean': 139301 * 2
    }

    t = 'mean'
    if 0:
        results = multi_gpu_benchmark(tests, frame_counts, frame_wh)
        multi_gpu_graphs(t, frame_counts, frame_wh, results)

    if 1:
        t = 'fight'
        all_results = {'fight': {'caffe': [450.6003510117739,
                                                                744.8901229071761,
                                                                1214.9085580870278],
                                            'flow': [35.26797046326607, 65.1234304140463, 111.91821397303859],
                                            'histogram': [817.7005547708027,
                                                                                   1676.5330527934939,
                                                                                   3309.0863111932586]}}
        multi_gpu_graphs(t, frame_counts, frame_wh, all_results)
    if 1:
        t = 'mean'
        results = {'mean': {'caffe': [1100.922914437792, 2188.3067699888497, 4350.245467315307],
                            'flow': [130.15578312203905, 239.4233822453851, 355.9739890240647],
                            'histogram': [3353.6737094160358,
                                          6694.3141921293845,
                                          12225.677026449643]}}
        multi_gpu_graphs(t, frame_counts, frame_wh, results)


BENCHMARKS = {
    'multi_gpu': multi_gpu_benchmark,
    'standalone': standalone_benchmarks,
    'scanner': scanner_benchmarks,
    'peak': peak_benchmarks,
}


def benchmark(args):
    test = args.test
    out_dir = args.output_directory
    if test == 'all':
        for name, fn in BENCHMARKS.iteritems():
            fn()
    else:
        fn = BENCHMARKS[test]
        fn()
