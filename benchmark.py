from __future__ import print_function
import os
import os.path
import time
import subprocess
import sys
import struct
import json
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
from scannerpy import Database, DeviceType, ScannerException, Job, BulkJob
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

#DB_PATH = '/tmp/scanner_db'
DB_PATH = '/n/scanner/apoms/dbs/bench_db'
DEBUG = False

CRISSY_NUM_CPUS = 16
CRISSY_NUM_GPUS = 4

GCE_NUM_CPUS = 16
GCE_NUM_GPUS = 8

def clear_filesystem_cache():
    os.system('sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"')


def make_db(master=None, workers=None, opts={}):
    config_path = opts['config_path'] if 'config_path' in opts else None
    db_path = opts['db_path'] if 'db_path' in opts else DB_PATH
    config = Config(config_path, db_path=db_path)
    #return Database(master='localhost:5001', workers=['localhost:5003'],
    #                config=config)
    return Database(master=master,
                    workers=workers,
                    debug=True if master is None else False,
                    config=config)


def run_trial(db, bulk_job, opts={}):
    print('Running trial...')
    # Clear cache
    scanner_opts = {'profiling': True}
    def add_opt(s):
        if s in opts:
            scanner_opts[s] = opts[s]
    add_opt('io_packet_size')
    add_opt('work_packet_size')
    add_opt('cpu_pool')
    add_opt('gpu_pool')
    add_opt('pipeline_instances_per_node')
    add_opt('tasks_in_queue_per_pu')

    force_cpu_decode = \
      opts['force_cpu_decode'] if 'force_cpu_decode' in opts else False
    no_pipelining = \
      opts['no_pipelining'] if 'no_pipelining' in opts else False
    start = now()
    success = True
    prof = None
    try:
        clear_filesystem_cache()
        if force_cpu_decode:
            os.environ['FORCE_CPU_DECODE'] = ''
        if no_pipelining:
            os.environ['NO_PIPELINING'] = ''
        out_table = db.run(bulk_job, force=True, **scanner_opts)
        if force_cpu_decode:
            del os.environ['FORCE_CPU_DECODE']
        if no_pipelining:
            del os.environ['NO_PIPELINING']
        prof = out_table[0].profiler() if isinstance(out_table, list) else out_table.profiler()
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
    return success, t, prof


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

def cpm_ablation_benchmark():
    # input_video = '/n/scanner/datasets/panoptic/160422_mafia2/vgaVideos/vga_01_01.mp4'
    input_video = '/n/scanner/datasets/movies/private/excalibur_1981.mp4'

    with make_db() as db:
        if not db.has_table(input_video):
            db.ingest_videos([(input_video, input_video)])

        descriptor = NetDescriptor.from_file(db, 'nets/cpm2.toml')
        cpm2_args = db.protobufs.CPM2Args()
        cpm2_args.scale = 368.0/480.0
        caffe_args = cpm2_args.caffe_args
        caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())

    def run(db, pipeline, k, opts):
        task_size = opts['task_size']
        work_item_size = opts['work_item_size']
        batch = opts['batch']
        input_op = db.table(input_video).as_op()

        def histogram():
            frame = input_op.range(0, 20000, task_size = task_size)
            histogram = db.ops.Histogram(frame=frame, device=DeviceType.GPU)
            return histogram

        def flow_gpu():
            frame = input_op.range(0, 10000, task_size = task_size)
            flow = db.ops.OpticalFlow(frame=frame, batch=batch, device=DeviceType.GPU)
            return db.ops.DiscardFrame(ignore=flow, device=DeviceType.GPU)

        def cpm():
            frame = input_op.range(0, 22500, task_size = task_size)
            frame_info = db.ops.InfoFromFrame(frame = frame)
            caffe_args.batch_size = 1
            cpm2_input = db.ops.CPM2Input(
                frame = frame,
                args = cpm2_args,
                device = DeviceType.GPU,
                batch = batch_size)
            cpm2_resized_map, cpm2_joints = db.ops.CPM2(
                cpm2_input = cpm2_input,
                args = cpm2_args,
                device = DeviceType.GPU,
                batch = batch_size)
            return db.ops.CPM2Output(
                cpm2_resized_map = cpm2_resized_map,
                cpm2_joints = cpm2_joints,
                original_frame_info = frame_info,
                args = cpm2_args,
                batch = batch_size)

        def caffe():
            # Caffe benchmark
            descriptor = NetDescriptor.from_file(db, 'nets/googlenet.toml')
            caffe_args = db.protobufs.CaffeArgs()
            caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())
            caffe_args.batch_size = batch

            frame = input_op.range(0, 20000, task_size = task_size)
            caffe_input = db.ops.CaffeInput(
                frame=frame,
                batch=batch,
                args=caffe_args,
                device=DeviceType.GPU)
            return db.ops.Caffe(
                caffe_frame=caffe_input,
                batch=batch,
                args=caffe_args,
                device=DeviceType.GPU)

        pipelines = {
            'histogram': histogram,
            'flow_gpu': flow_gpu,
            'cpm': cpm,
            'caffe': caffe
        }

        job = Job(columns = [pipelines[pipeline]()], name = 'example_poses')
        success, total, prof = run_trial(db, job, opts)
        prof.write_trace('{}_{}.trace'.format(pipeline, k))
        return total

    timings = {}
    for pipeline in ['flow_gpu']:
        options = {
            'pipeline_instances_per_node': 1,
            'no_pipelining': True,
            'work_item_size': 1,
            'batch': 1,
            'task_size': 1000,
            'force_cpu_decode': True,
            'tasks_in_queue_per_pu': 1,
            'io_item_size': 1000
        }

        values = [
            ('pipeline_instances_per_node', 4),
            ('no_pipelining', False),
            ('force_cpu_decode', False),
            ('gpu_pool', '4425M'),
            ('work_item_size', 96),
            ('batch', 96)]
        # if pipeline == 'flow_gpu':
        #     del values[4:6]

        timings[pipeline] = []
        with make_db() as db:
            timings[pipeline] = \
              [('baseline', run(db, pipeline, 'baseline', options))]
        pprint(timings)

        for (k, v) in values:
            options[k] = v
            if k == 'no_pipelining':
                options['tasks_in_queue_per_pu'] = 4

            if k == 'pipeline_instances_per_node':
                if pipeline == 'caffe':
                    options['pipeline_instances_per_node'] = 1

            if pipeline == 'caffe':
                nodes = ['ocean.pdl.local.cmu.edu:500{:d}'.format(i) for i in range(5)]
                with make_db(master=nodes[0], workers=nodes[1:]) as db:
                    t = run(db, pipeline, k, options)
            else:
                with make_db() as db:
                    t = run(db, pipeline, k, options)
            timings[pipeline].append((k, t))
            pprint(timings)


# TODO: re-encode video from stride
def video_encoding_benchmark_2():
    input_videos = [
        '/n/scanner/datasets/movies/private/excalibur_1981.mp4',
        '/n/scanner/datasets/movies/private/toy_story_3_2010.mp4',
        '/n/scanner/datasets/movies/private/interstellar_2014.mp4']

    with Database() as db:
        def decode(ts, fn, image = False, device = DeviceType.CPU, profile = None):
            jobs = []
            for t in ts:
                t = db.table(t)
                frame = fn(t.as_op())
                if image:
                    frame = db.ops.ImageDecoder(
                        img = frame,
                        image_type = db.protobufs.ImageDecoderArgs.JPEG,
                        batch = 10)
                dummy = db.ops.DiscardFrame(ignore = frame, device = device, batch = 10)
                job = Job(columns = [dummy], name = 'ignore_{}'.format(t.name()))
                jobs.append(job)

            success, t, prof = run_trial(db, jobs)
            assert(success)
            if profile is not None:
                prof.write_trace('{}.trace'.format(profile))

            return t

        for input_video in input_videos:
            if not db.has_table(input_video):
                print('Ingesting baseline {}'.format(input_video))
                db.ingest_videos([(input_video, input_video)], force=True)

        _, f = next(db.table(input_videos[0]).load(['frame'], rows=[0]))
        [input_height, input_width, _] = f[0].shape

        times = {
            'vid_cpu': {},
            'vid_gpu': {},
            'vid_smallgop_cpu': {},
            'vid_smallgop_gpu': {},
            'vid_strided_cpu': {},
            'vid_strided_gpu': {},
            'img_cpu': {},
        }

        sizes = {
            'vid': {},
            'vid_strided': {},
            'vid_smallgop': {},
            'img': {}
        }

        def table_size(table):
            tid = db.table(table).id()
            path = '{}/tables/{}'.format(db._db_path, tid)
            return subprocess.check_output(['du', '-bh', path]).split('\t')[0]

        for scale in [1, 4]:
            width = (input_width / scale) // 2 * 2
            height = (input_height / scale) // 2 * 2

            strides = [1, 2, 4, 8, 16, 32, 64]

            vid_names = ['{}_{}'.format(input_video, scale)
                         for input_video in input_videos]
            vid_smallgop_names = ['{}_{}_smallgop'.format(input_video, scale)
                                  for input_video in input_videos]
            vid_strided_names = [['{}_{}_strided_{}'.format(input_video, scale,
                                                            stride)
                                  for stride in strides]
                                     for input_video in input_videos]
            img_names = ['{}_{}_img'.format(input_video, scale)
                         for input_video in input_videos]

            sizes['vid'][scale] = []
            sizes['vid_smallgop'][scale] = []
            sizes['vid_strided'][scale] = {}
            for stride in strides:
                sizes['vid_strided'][scale][stride] = []
            sizes['img'][scale] = []

            for (input_video, vid_name) in zip(input_videos, vid_names):
                if not db.has_table(vid_name):
                    print('Resizing baseline to {}'.format(vid_name))
                    frame = db.table(input_video).as_op().all(task_size=250)
                    resized = db.ops.Resize(
                        frame = frame, width = width, height = height,
                        device = DeviceType.CPU)
                    job = Job(columns = [resized.compress_video()], name = vid_name)
                    t = now()
                    out = db.run(job, force = True,
                                 pipeline_instances_per_node=16)
                    print('{:.3f}'.format(now() - t))
                sizes['vid'][scale].append(table_size(vid_name))

            for (vid_name, names) in zip(vid_names, vid_strided_names):
                for (stride, vid_strided_name) in zip(strides, names):
                    if not db.has_table(vid_strided_name):
                        print('Re encoding with stride for {}'.format(vid_strided_name))
                        frame = db.table(vid_name).as_op().strided(stride, task_size=200)
                        frame = db.ops.PassthroughFrame(frame = frame)
                        job = Job(columns = [
                            frame.compress_video()],
                            name = vid_strided_name)
                        t = now()
                        out = db.run(job, force=True, pipeline_instances_per_node=16)
                        print('{:.3f}'.format(now() - t))
                    sizes['vid_strided'][scale][stride].append(table_size(vid_strided_name))

            for (vid_name, vid_smallgop_name) in zip(vid_names, vid_smallgop_names):
                if not db.has_table(vid_smallgop_name):
                    print('Re encoding with small gop for {}'.format(vid_smallgop_name))
                    frame = db.table(vid_name).as_op().all(task_size=200)
                    frame = db.ops.PassthroughFrame(frame = frame)
                    job = Job(columns = [
                        frame.compress_video(keyframe_distance=24)],
                        name = vid_smallgop_name)
                    t = now()
                    out = db.run(job, force=True, pipeline_instances_per_node=10)
                    print('{:.3f}'.format(now() - t))
                sizes['vid_smallgop'][scale].append(table_size(vid_smallgop_name))

            for (img_name, vid_name) in zip(img_names, vid_names):
                if not db.has_table(img_name):
                    print('Dumping frames to {}'.format(img_name))
                    frame = db.table(vid_name).as_op().all(task_size=250)
                    img = db.ops.ImageEncoder(frame = frame)
                    job = Job(columns = [img], name = img_name)
                    t = now()
                    db.run(job, force = True,
                           pipeline_instances_per_node=16)
                    print('{:.3f}'.format(now() - t))
                sizes['img'][scale].append(table_size(img_name))

            pprint(sizes)

            for k in times:
                times[k][scale] = {}

            for j, stride in enumerate(strides):
                if stride == 1:
                    fn = lambda item: lambda t: t.all(task_size=item)
                else:
                    fn = lambda item: lambda t: t.strided(stride, task_size=item)

                # t = decode(vid_names, fn(1000)) #profile='vid_cpu_{}'.format(scale))
                # times['vid_cpu'][scale][stride] = t

                # t = decode(vid_names, fn(1000), device = DeviceType.GPU)  #profile='vid_gpu_{}'.format(scale))
                # times['vid_gpu'][scale][stride] = t

                # t = decode(vid_smallgop_names, fn(1000)) #profile='vid_cpu_{}'.format(scale))
                # times['vid_smallgop_cpu'][scale][stride] = t

                # t = decode(vid_smallgop_names, fn(1000), device = DeviceType.GPU)  #profile='vid_gpu_{}'.format(scale))
                # times['vid_smallgop_gpu'][scale][stride] = t

                t = decode([l[j] for l in vid_strided_names], lambda t: t.all(task_size=1000),
                               device = DeviceType.CPU)
                times['vid_strided_cpu'][scale][stride] = t

                t = decode([l[j] for l in vid_strided_names], lambda t: t.all(task_size=1000),
                               device = DeviceType.GPU)
                times['vid_strided_gpu'][scale][stride] = t

                # t = decode(img_names, fn(1000), image = True) #profile = 'img_cpu_{}'.format(stride))
                # times['img_cpu'][scale][stride] = t

                pprint(times)


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


def standalone_benchmark(video, tests):
    video_frames = video['frames']
    output_dir = '/tmp/standalone'
    test_output_dir = '/tmp/standalone_outputs'
    paths_file = os.path.join(output_dir, 'paths.txt')

    def write_paths(paths):
        with open(paths_file, 'w') as f:
            f.write(paths[0])
            for p in paths[1:]:
                f.write('\n' + p)

    def run_standalone_trial(paths_file, operation, frames, stride,
                             decode_type, decode_args):
        print('Running standalone trial: {}, {}'.format(
            paths_file,
            operation))
        clear_filesystem_cache()
        current_env = os.environ.copy()
        start = time.time()
        program_path = os.path.join(
            COMPARISON_DIR, 'build/standalone/standalone_comparison')
        cmd = [program_path,
               '--paths_file', paths_file,
               '--operation', operation,
               '--frames', str(frames),
               '--stride', str(stride),
               '--decode_type', decode_type,
               '--decode_args', decode_args]
        print(' '.join(cmd))
        p = subprocess.Popen(cmd,
                             env=current_env,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
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

    video_path = video['path']
    table_name = 'test_video'
    with make_db() as db:
        [table], f = db.ingest_videos([(table_name, video_path)], force=True)
        assert(len(f) == 0)
        table_id = table.id()
        column_id = -1
        for c in table.column_names():
            if c == 'frame':
                column_id = table.column(c).id()
        assert column_id != -1

    bmp_template = "ffmpeg -i {input} -start_number 0 {output}/frame%07d.bmp"
    jpg_template = "ffmpeg -i {input} -start_number 0 {output}/frame%07d.jpg"

    os.system('rm -rf {}'.format(output_dir))
    os.system('mkdir {}'.format(output_dir))

    video_samplings = video['samplings']
    path = video['path']
    run_paths = []
    base = os.path.basename(path)
    run_path = os.path.join(output_dir, base)
    os.system('cp {} {}'.format(path, run_path))
    run_paths.append(run_path)
    write_paths(run_paths)

    results = {}
    for test in tests:
        name = test['name']

        test_type = name

        sampling = video_samplings[test['sampling']]
        results[name] = []

        frames = video_frames
        stride = 1
        sampling_type = sampling[0]
        decode_type = 'all'
        decode_args = ''
        if sampling_type == 'all':
            frames = video_frames
        elif sampling_type == 'range':
            frames = 0
            for s, e in sampling[1]:
                frames += (e - s)
            decode_type = 'range'
            decode_args = ','.join(['{:d}:{:d}'.format(s, e)
                                    for s, e in sampling[1]])
        elif sampling_type == 'strided':
            stride = sampling[1]
            frames = video_frames / stride
            decode_type = 'strided'
        elif sampling_type == 'gather':
            decode_type = 'gather'
            gather_path = '/tmp/gather_path.txt'
            os.system('rm -f {:s}'.format(gather_path))
            with open(gather_path, 'w') as f:
                for k in sampling[1]:
                    f.write(str(k) + '\n')
            decode_args = gather_path
            frames = len(sampling[1])
        elif sampling_type == 'keyframe':
            decode_type = 'gather'
            keyframes_list = table.column('frame').keyframes()
            gather_path = '/tmp/gather_path.txt'
            os.system('rm -f {:s}'.format(gather_path))
            with open(gather_path, 'w') as f:
                for k in keyframes_list:
                    f.write(str(k) + '\n')
            decode_args = gather_path
            frames = len(keyframes_list)


        print('decode_args', decode_args)
        os.system('rm -rf {}'.format(test_output_dir))
        os.system('mkdir -p {}'.format(test_output_dir))
        total, timings = run_standalone_trial(paths_file, test_type,
                                              video_frames, stride,
                                              decode_type, decode_args)
        results[name].append({'results': (total, timings),
                              'frames': frames})

    print(results)
    return results


def scanner_benchmark(video, tests):
    # Histogram benchmarks
    def decode_pipeline(device):
        def fn(frame):
            return db.ops.DiscardFrame(ignore=frame, device=device)
        return fn

    decode_cpu = decode_pipeline(device=DeviceType.CPU)
    decode_gpu = decode_pipeline(device=DeviceType.GPU)

    def hist_pipeline(device):
        def fn(frame):
            return db.ops.Histogram(frame=frame, batch=128, device=device)
        return fn

    hist_cpu = hist_pipeline(device=DeviceType.CPU)
    hist_gpu = hist_pipeline(device=DeviceType.GPU)

    # Optical flow benchmarks
    def of_cpu_pipeline(frame):
        dt = DeviceType.CPU
        flow = db.ops.OpticalFlow(frame=frame, device=dt)
        return db.ops.DiscardFrame(ignore=flow, device=dt)

    def of_gpu_pipeline(frame):
        dt = DeviceType.GPU
        flow = db.ops.OpticalFlow(frame=frame, batch=8, device=dt)
        return db.ops.DiscardFrame(ignore=flow, device=dt)

    of_cpu = of_cpu_pipeline
    of_gpu = of_gpu_pipeline

    # Caffe benchmark
    with make_db() as db:
        descriptor = NetDescriptor.from_file(db, 'nets/googlenet.toml')
    caffe_args = db.protobufs.CaffeArgs()
    caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())
    caffe_args.batch_size = 96

    def caffe_pipeline(device):
        def fn(frame):
            caffe_input = db.ops.CaffeInput(
                frame=frame,
                batch=96,
                args=caffe_args,
                device=device)
            return db.ops.Caffe(
                caffe_frame=caffe_input,
                batch=96,
                args=caffe_args,
                device=device)
        return fn

    caffe_cpu = caffe_pipeline(device=DeviceType.CPU)
    caffe_gpu = caffe_pipeline(device=DeviceType.GPU)

    # Multi-view stereo benchmark

    operations = {
        'decode_cpu': decode_cpu,
        'decode_gpu': decode_gpu,
        'stride_cpu': decode_cpu,
        'stride_gpu': decode_gpu,
        'gather_cpu': decode_cpu,
        'gather_gpu': decode_gpu,
        'range_cpu': decode_cpu,
        'range_gpu': decode_gpu,
        'keyframe_cpu': decode_cpu,
        'keyframe_gpu': decode_gpu,
        # 'join_cpu': join_cpu,
        # 'join_gpu': join_gpu,

        'histogram_cpu': hist_cpu,
        'histogram_gpu': hist_gpu,
        'flow_cpu': of_cpu,
        'flow_gpu': of_gpu,
        'caffe': caffe_gpu,
    }

    os.system('rm -rf {}'.format(DB_PATH))

    # ingest data
    video_samplings = video['samplings']
    video_path = video['path']
    total_frames = video['frames']
    table_name = 'test_video'
    with make_db() as db:
        _, f = db.ingest_videos([(table_name, video_path),
                                 (table_name + '_2', video_path),
                                 (table_name + '_3', video_path),
                                 (table_name + '_4', video_path)], force=True)
        assert(len(f) == 0)

    results = {}
    for test in tests:
        name = test['name']
        is_gpu = True if name[-3:] == 'gpu' else False
        if name == 'caffe':
            is_gpu = True

        ops = None
        for o, pipeline in operations.iteritems():
            if name.startswith(o):
                ops = pipeline
                break
        assert(ops is not None)

        settings = test['scanner_settings']
        sampling = video_samplings[test['sampling']]

        results[name] = []

        # Parse sampling
        io_packet_size = settings['io_packet_size']
        # Parse settings
        opts = settings
        if 'work_packet_size' not in settings:
            work_packet_size = 256 if is_gpu else 64
            if work_packet_size > io_packet_size:
                work_packet_size = io_packet_size
            opts['work_packet_size'] = work_packet_size
        else:
            opts['work_packet_size'] = settings['work_packet_size']
        opts['io_packet_size'] = io_packet_size
        if 'nodes' in opts:
            master = opts['nodes'][0]
            workers = opts['nodes'][1:]
        else:
            master = None
            workers = None
        with make_db(master=master, workers=workers) as db:
            print('Running {:s}'.format(name))

            # Instantiate pipeline
            frame_input = db.ops.FrameInput()
            sampled_frame_input = frame_input.sample()
            output_op = db.ops.Output(columns=[ops(sampled_frame_input)])

            # Create jobs with desired sampling
            sampling_type = sampling[0]
            print(sampling)
            jobs = []
            frames = 0
            for t_name in [table_name, table_name + '_2',
                           table_name + '_3', table_name + '_4']:
                frame_column = db.table(t_name).column('frame')
                if sampling_type == 'all':
                    sampling_spec = db.sampler.all()
                    frames += total_frames
                elif sampling_type == 'strided':
                    sampling_spec = db.sampler.strided(sampling[1])
                    frames += total_frames / sampling[1]
                elif sampling_type == 'gather':
                    sampling_spec = db.sampler.gather(sampling[1])
                    frames += len(sampling[1])
                elif sampling_type == 'range':
                    sampling_spec = db.sampler.ranges(sampling[1])
                    frames += sum(e - s for s, e in sampling[1])
                elif sampling_type == 'keyframe':
                    keyframes = frame_column.keyframes()
                    print(t_name)
                    print(len(keyframes))
                    sampling_spec = db.sampler.gather(keyframes)
                    frames += len(keyframes)
                else:
                    print('Not a valid sampling type:', sampling_type)
                    exit(1)

                job = Job(op_args={
                    frame_input: frame_column,
                    sampled_frame_input: sampling_spec,
                    output_op: t_name + '_dummy_output',
                })
                jobs.append(job)
            bulk_job = BulkJob(jobs=jobs, output=output_op)
            success, total, prof = run_trial(db, bulk_job, opts)
            assert(success)
            stats = prof.statistics()
            prof.write_trace(name + '.trace')
            results[name].append({'results': (total, stats), 'frames': frames})

    print(video)
    pprint(results)
    return results


def peak_benchmark(video, tests):
    test_output_dir = '/tmp/peak_outputs'

    def run_peak_trial(list_path, op, decode_type, decode_args, db_path,
                       scanner_video_args, width, height,
                       decoders, evaluators):
        print('Running peak trial: {}'.format(op))
        clear_filesystem_cache()
        current_env = os.environ.copy()
        start = time.time()
        program_path = os.path.join(
            COMPARISON_DIR, 'build/peak/peak_comparison')

        cmd = ' '.join([
            program_path,
            '--video_list_path', list_path,
            '--operation', op,
            '--decode_type', decode_type,
            '--decode_args', decode_args,
            '--db_path', db_path,
            '--scanner_video_args', scanner_video_args,
            '--decoder_count', str(decoders),
            '--eval_count', str(evaluators),
            '--width', str(width),
            '--height', str(height)])
        print(cmd)
        p = subprocess.Popen([
            program_path,
            '--video_list_path', list_path,
            '--operation', op,
            '--decode_type', decode_type,
            '--decode_args', decode_args,
            '--db_path', db_path,
            '--scanner_video_args', scanner_video_args,
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
            for f in files:
                paths.append(f)
            #total_frames += count_frames('{}/{}'.format(videos_dir, video))

        #print('Total frames', total_frames)
        print('num paths', len(paths))
        return paths

    total_frames = video['frames']
    width = video['width']
    height = video['height']
    paths = [video['path']]

    os.system('rm -rf {}'.format(DB_PATH))

    # Ingest video for scanner
    video_path = video['path']
    table_name = 'test_video'
    with make_db() as db:
        [table], f = db.ingest_videos([(table_name, video_path)], force=True)
        assert(len(f) == 0)
        table_id = table.id()
        column_id = -1
        for c in table.column_names():
            if c == 'frame':
                column_id = table.column(c).id()
        assert column_id != -1

    scanner_video_args = '{:d}:{:d}'.format(table_id, column_id)

    all_results = {}
    for test in tests:
        test_name = test['name']
        sampling = video['samplings'][test['sampling']]

        decode_type = 'all'
        decode_args = ''

        all_results[test_name] = {}

        settings = test['peak_settings']
        op = test_name
        dec = settings['decoders']
        ev = settings['evaluators']
        tt = settings['tt']
        seg = settings['seg']

        # video
        os.system('rm -rf {}'.format(test_output_dir))
        os.system('mkdir -p {}'.format(test_output_dir))

        frames = total_frames
        #if op == 'flow_cpu' or op == 'flow_gpu':
        #    frames = 8632

        # for opencv or ffmpeg
        video_paths = split_videos(paths, '/tmp/peak_videos', tt, seg)
        os.system('rm -f /tmp/peak_videos.txt')
        with open('/tmp/peak_videos.txt', 'w') as f:
            for p in video_paths:
                f.write(p + '\n')

        # for scanner

        if sampling[0] == 'strided':
            decode_type = 'strided'
            stride = sampling[1]
            decode_args = str(stride)
            frames = total_frames / stride
        elif sampling[0] == 'range':
            decode_type = 'range'
            frames = 0
            for s, e in sampling[1]:
                frames += (e - s)
            decode_args = ','.join(['{:d}:{:d}'.format(s, e)
                                    for s, e in sampling[1]])
        elif sampling[0] == 'keyframe':
            decode_type = 'gather'
            keyframes_list = table.column('frame').keyframes()
            gather_path = '/tmp/gather_path.txt'
            os.system('rm -f {:s}'.format(gather_path))
            with open(gather_path, 'w') as f:
                for k in keyframes_list:
                    f.write(str(k) + '\n')
            decode_args = gather_path
            frames = len(keyframes_list)

        all_results[test_name] = [{
            'results': run_peak_trial('/tmp/peak_videos.txt', op,
                                      decode_type,
                                      decode_args,
                                      DB_PATH,
                                      scanner_video_args,
                                      width,
                                      height,
                                      dec, ev),
            'frames': frames}]

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
        task_size = settings['task_size']
        sampling_type = sampling[0]
        print(sampling)
        if sampling_type == 'all':
            sampled_input = db.sampler().all(collection, task_size=task_size)
            frames = total_frames
        elif sampling_type == 'strided':
            sampled_input = db.sampler().strided(collection,
                                                 sampling[1],
                                                 task_size=task_size)
            frames = total_frames / sampling[1]
        elif sampling_type == 'gather':
            sampled_input = [db.sampler().gather(collection,
                                                 sampling[1],
                                                 task_size=task_size)]
            frames = len(sampling[1])
        elif sampling_type == 'range':
            sampled_input = db.sampler().ranges(collection,
                                                sampling[1],
                                                task_size=task_size)
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

    ltests = []
    for t in tests:
        x = t.copy()
        x['sampling'] = large_samplings[t['sampling']]
        ltests.append(x)

    standalone_results = standalone_benchmark(large_video, large_video_frames, ltests)
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


def surround360_single_node_benchmark():
    cmd_args = (
        '--flow_alg pixflow_search_20 ' +
        '--rig_json_file ~/temp/geometric_calibration/camera_rig.json ' +
        '--quality 8k ' +
        '--start_frame 0 ' +
        '--end_frame {num_frames:d} ' +
        '--surround360_render_dir {surround360_dir:s} ' +
        '--cores {cores:d} ' +
        '--root_dir ~/datasets/surround360/palace3')
        #'--root_dir /n/scanner/datasets/surround360/palace3')
    num_frames = 150
    #num_frames = 2
    surround360_dir = '/h/apoms/repos/Surround360/surround360_render'

    base_cmd = os.path.join(surround360_dir,
                            'scripts/batch_process_video.py')
    scanner_cmd = os.path.join(surround360_dir,
                               'scripts/scanner_process_video.py')
    cmd_template = 'taskset -c {cpu_list:s} python {cmd:s} {args:s}'

    cpus = [2**i for i in range(1, 7)]
    #cpus = [2**i for i in range(5, 6)]
    base_tests = []
    scanner_tests = []
    for c in cpus:
        # Base test
        cpu_list = ','.join([str(d) for d in range(c)])
        cmd = cmd_template.format(
            cpu_list=cpu_list,
            cmd=base_cmd,
            args=cmd_args.format(num_frames=num_frames - 1,
                                 surround360_dir=surround360_dir,
                                 cores=c))
        print('Running surrround360 base, {:d} cores'.format(c))
        clear_filesystem_cache()
        current_env = os.environ.copy()
        start = time.time()
        print(cmd)
        p = subprocess.Popen(
            cmd, env=current_env, stdout=DEVNULL, stderr=subprocess.STDOUT,
            shell=True)
        pid, rc, ru = os.wait4(p.pid, 0)
        elapsed = time.time() - start
        if rc != 0:
            print('Surround360 base FAILED after {:.3f}s'.format(elapsed))
            elapsed = -1
        else:
            print('Surround360 base succeeded, took {:.3f}s'.format(elapsed))
        base_tests.append(elapsed)

        # Scanner test
        cmd = cmd_template.format(
            cpu_list=cpu_list,
            cmd=scanner_cmd,
            args=cmd_args.format(num_frames=num_frames - 1,
                                 surround360_dir=surround360_dir,
                                 cores=c))
        print('Running surrround360 scanner, {:d} cores'.format(c))
        clear_filesystem_cache()
        start = now()
        elapsed = now() - start
        current_env = os.environ.copy()
        start = time.time()
        print(cmd)
        p = subprocess.Popen(
            cmd, env=current_env, stdout=DEVNULL, stderr=subprocess.STDOUT,
            shell=True)
        pid, rc, ru = os.wait4(p.pid, 0)
        elapsed = time.time() - start
        if rc != 0:
            print('Surround360 scanner FAILED after {:.3f}s'.format(elapsed))
            elapsed = -1
        else:
            print('Surround360 scanner succeeded, took {:.3f}s'.format(elapsed))
        scanner_tests.append(elapsed)

    print('CPUs', cpus)
    print('Frames', num_frames)
    print('Base')
    print(base_tests)
    print('Scanner')
    print(scanner_tests)
    graph.surround360_single_node_graph(num_frames, cpus,
                                        base_tests, scanner_tests)


def make_video_interval(total_frames):
    # A range 100th of the video every 10th of the video
    return [
        (f - (total_frames / 100), f)
        for f in range(total_frames / 10, total_frames + 1,
                       total_frames / 20)]

DATA_PREFIX = '/n/scanner/'
SMALL_FC = 139301
LARGE_FC = 202525
VIDEOS = {
    'small': {
        'path': os.path.join(DATA_PREFIX,
                             'datasets/movies/private/meanGirls.mp4'),
        'frames': 139301,
        'width': 640,
        'height': 480,
        'samplings': {
            'all': ('all',),
            'strided_short': ('strided', 24),
            'strided_long': ('strided', 500),
            'gather': ('gather', None),
            'range': ('range', make_video_interval(SMALL_FC)),
            'hist_cpu_all': ('range', [[0, SMALL_FC / 4]]),
            'caffe_all': ('range', [[0, SMALL_FC / 4]]),
            'flow_cpu_all': ('range', [[0, SMALL_FC / 100]]),
            'flow_gpu_all': ('range', [[0, SMALL_FC / 20]]),
            'keyframe': ('keyframe',),
        }
    },
    'large': {
        'path': os.path.join(DATA_PREFIX,
                             'datasets/movies/private/excalibur_1981.mp4'),
        'frames': 202525,
        'width': 1920,
        'height': 1080,
        'samplings': {
            'all': ('all',),
            'strided_short': ('strided', 24),
            'strided_long': ('strided', 500),
            'gather': ('gather', None),
            'range': ('range', make_video_interval(LARGE_FC)),
            'hist_cpu_all': ('range', [[0, (2**17 + 2**16)/3]]),
            'caffe_all': ('range', [[0, LARGE_FC / 2]]),
            'flow_cpu_all': ('range', [[0, LARGE_FC / 48]]),
            #'flow_cpu_all': ('range', [[0, 100]]),
            #'flow_gpu_all': ('range', [[0, LARGE_FC / 10]]),
            'flow_gpu_all': ('range', [[0, 10240 * 8]]),
            'keyframe': ('keyframe',),
        }
    }
}


def decode_sol():
    video = VIDEOS['large']

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

    video_path = video['path']
    temp_video_path = '/tmp/vid'

    os.system('rm -rf {}'.format(DB_PATH))
    os.system('cp {} {}'.format(video_path, temp_video_path))

    test_name = 'large'
    frame_count = video['frames']
    all_results = {}
    all_results[test_name] = {}

    # Scanner decode
    with make_db() as db:
        [table], failed_vids = db.ingest_videos(
            [('decode_test', temp_video_path)], force=True)
        assert len(failed_vids) == 0

    with make_db() as db:
        def decode_pipeline(frame, device, batch):
            return db.ops.DiscardFrame(ignore=frame,
                                       device=device,
                                       batch=batch)

        frame = table.as_op().all(task_size=768)
        job = Job(columns=[decode_pipeline(frame, DeviceType.CPU, batch=192)],
                  name='dummy')
        succ, total, prof = run_trial(db, job)
        assert succ
        prof.write_trace('decode_cpu.trace')
        all_results[test_name]['scanner_cpu'] = total

    with make_db() as db:
        def decode_pipeline(frame, device, batch):
            return db.ops.DiscardFrame(ignore=frame,
                                       device=device,
                                       batch=batch)
        frame = table.as_op().all(task_size=8192)
        job = Job(columns=[decode_pipeline(frame, DeviceType.GPU, batch=256)],
                  name='dummy')
        succ, total, prof = run_trial(db, job,
                                      {'gpu_pool': '2G',
                                       'pipeline_instances_per_node': 1})
        assert succ
        prof.write_trace('decode_gpu.trace')
        all_results[test_name]['scanner_gpu'] = total

    # OCV decode
    total, _ = run_ocv_trial('cpu', temp_video_path)
    all_results[test_name]['opencv_cpu'] = total

    total, _ = run_ocv_trial('gpu', temp_video_path)
    all_results[test_name]['opencv_gpu'] = total

    # FFMPEG CPU decode
    start_time = time.time()
    run_cmd(ffmpeg_cpu_template, {'path': temp_video_path})
    all_results[test_name]['ffmpeg_cpu'] = time.time() - start_time

    # FFMPEG GPU decode
    start_time = time.time()
    run_cmd(ffmpeg_gpu_template, {'path': temp_video_path})
    all_results[test_name]['ffmpeg_gpu'] = time.time() - start_time

    print('Decode test on ', video_path)
    print("{:10s} | {:6s} | {:7s}".format('Type', 'Total', 'FPS'))
    for ty, total in all_results[test_name].iteritems():
        print("{:10s} | {:6.2f} | {:7.2f}".format(
            ty, total, frame_count / total))

    print(all_results)
    return all_results


def write_results(path, results):
    with open(path, 'w') as f:
        json.dump(results, f)

def read_results(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    else:
        return {}

def run_full_comparison(prefix, video, tests, recalculate=True):
    standalone_results = read_results(
        '{:s}_standalone_results.json'.format(prefix))
    scanner_results = read_results('{:s}_scanner_results.json'.format(prefix))
    peak_results = read_results('{:s}_peak_results.json'.format(prefix))

    write = True
    if recalculate:
        #results = standalone_benchmark(video, tests)
        #for k, v in results.iteritems():
        #    standalone_results[k] = v
        if write:
            write_results('{:s}_standalone_results.json'.format(prefix),
                          standalone_results)

        #results = scanner_benchmark(video, tests)
        #for k, v in results.iteritems():
        #    scanner_results[k] = v
        if write:
            write_results('{:s}_scanner_results.json'.format(prefix),
                          scanner_results)

        #results = peak_benchmark(video, tests)
        #for k, v in results.iteritems():
        #    peak_results[k] = v
        if write:
            write_results('{:s}_peak_results.json'.format(prefix),
                          peak_results)


    return {'baseline': standalone_results,
            'scanner': scanner_results,
            'peak': peak_results}

##
def data_loading_benchmarks():
    tests = [
        {'name': 'decode_cpu',
         'sampling': 'all',
         'scanner_settings': {
             'io_packet_size': 2048,
             'cpu_pool': '32G',
             'pipeline_instances_per_node': 16
         },
         'peak_settings': {
             'decoders': 16,
             'evaluators': 1,
             'tt': '',
             'seg': '5',
         }},
        {'name': 'decode_gpu',
         'sampling': 'all',
         'scanner_settings': {
             'io_packet_size': 8192,
             'gpu_pool': '2G',
             'pipeline_instances_per_node': 1
         },
         'peak_settings': {
             'decoders': 1,
             'evaluators': 1,
             'tt': '',
             'seg': '5',
         }},
        {'name': 'stride_cpu',
         'sampling': 'strided_short',
         'scanner_settings': {
             'io_packet_size': 512,
             'cpu_pool': '32G',
             'pipeline_instances_per_node': 16
         },
         'peak_settings': {
             'decoders': 16,
             'evaluators': 1,
             'tt': '',
             'seg': '5',
         }},
        {'name': 'stride_gpu',
         'sampling': 'strided_short',
         'scanner_settings': {
             'io_packet_size': 2048,
             'gpu_pool': '2G',
             'pipeline_instances_per_node': 1
         },
         'peak_settings': {
             'decoders': 1,
             'evaluators': 1,
             'tt': '',
             'seg': '5',
         }},
        {'name': 'gather_cpu',
         'sampling': 'strided_long',
         'scanner_settings': {
             'cpu_pool': '32G',
             'io_packet_size': 1,
             'pipeline_instances_per_node': 16
         },
         'peak_settings': {
             'decoders': 16,
             'evaluators': 1,
             'tt': '',
             'seg': '5',
         }},
        {'name': 'gather_gpu',
         'sampling': 'strided_long',
         'scanner_settings': {
             'gpu_pool': '4G',
             'io_packet_size': 1,
             'pipeline_instances_per_node': 1
         },
         'peak_settings': {
             'decoders': 1,
             'evaluators': 1,
             'tt': '',
             'seg': '5',
         }},
        {'name': 'range_cpu',
         'sampling': 'range',
         'scanner_settings': {
             'io_packet_size': 1024,
             'cpu_pool': '32G',
             'pipeline_instances_per_node': 16
         },
         'peak_settings': {
             'decoders': 16,
             'evaluators': 1,
             'tt': '',
             'seg': '5',
         }},
        {'name': 'range_gpu',
         'sampling': 'range',
         'scanner_settings': {
             'io_packet_size': 2048,
             'gpu_pool': '2g',
             'pipeline_instances_per_node': 1
         },
         'peak_settings': {
             'decoders': 1,
             'evaluators': 1,
             'tt': '',
             'seg': '5',
         }},
        {'name': 'keyframe_cpu',
         'sampling': 'keyframe',
         'scanner_settings': {
             'work_packet_size': 1,
             'io_packet_size': 256,
             'cpu_pool': '32G',
             'pipeline_instances_per_node': 16
         },
         'peak_settings': {
             'decoders': 16,
             'evaluators': 1,
             'tt': '',
             'seg': '5',
         }},
        {'name': 'keyframe_gpu',
         'sampling': 'keyframe',
         'scanner_settings': {
             'work_packet_size': 1,
             'io_packet_size': 1,
             'gpu_pool': '2G',
             'pipeline_instances_per_node': 1
         },
         'peak_settings': {
             'decoders': 1,
             'evaluators': 1,
             'tt': '',
             'seg': '5',
         }},
    ]
    tests = [
        {'name': 'gather_cpu',
         'sampling': 'strided_long',
         'scanner_settings': {
             'cpu_pool': '32G',
             'io_packet_size': 1,
             'pipeline_instances_per_node': 16
         },
         'peak_settings': {
             'decoders': 16,
             'evaluators': 1,
             'tt': '',
             'seg': '5',
         }},
    ]
    tests = [
        {'name': 'gather_cpu',
         'sampling': 'gather',
         'scanner_settings': {
             'cpu_pool': '32G',
             'io_packet_size': 1,
             'pipeline_instances_per_node': 16
         },
         'peak_settings': {
             'decoders': 16,
             'evaluators': 1,
             'tt': '',
             'seg': '5',
         }},
        {'name': 'gather_gpu',
         'sampling': 'gather',
         'scanner_settings': {
             'gpu_pool': '4G',
             'io_packet_size': 1,
             'pipeline_instances_per_node': 1
         },
         'peak_settings': {
             'decoders': 1,
             'evaluators': 1,
             'tt': '',
             'seg': '5',
         }},
    ]
    # decode
    # stride
    # gather
    # range
    # join

    video_name = 'large'

    video_data = VIDEOS[video_name]

    # Generate gather list
    random.seed(1234)
    np.random.seed(1234)

    total_frames = video_data['frames']
    percentage = 0.0025
    frames_to_select = int(percentage * total_frames)
    gather_list = np.random.choice(total_frames, frames_to_select,
                                   replace=False)
    gather_list.sort()
    print('gather list', gather_list)
    video_data['samplings']['gather'] = ('gather', gather_list)

    recalculate = True
    results = run_full_comparison('loading', video_data, tests,
                                  recalculate)
    pprint(results)

    ops = ['decode_cpu',
           'stride_cpu',
           'range_cpu',
           'keyframe_cpu',
           'gather_cpu',

           'decode_gpu', 
           'stride_gpu',
           'range_gpu',
           'keyframe_gpu',
           'gather_gpu',]
    labels = ['STRIDE-1',
              'STRIDE-24',
              'RANGE',
              'KEYFRAME',
              'GATHER',

              'STRIDE-1',
              'STRIDE-24',
              'RANGE',
              'KEYFRAME',
              'GATHER']

    ticks = [1, 5, 10, 15, 20]
    graph.comparison_graphs('loading_{:s}'.format(video_name),
                            VIDEOS[video_name],
                            ops,
                            labels,
                            results['baseline'],
                            results['scanner'],
                            results['peak'],
                            ticks=ticks)


##
def micro_apps_benchmarks():
    # histogram
    # depth from stereo?
    # dnn
    # optical flow
    tests = [
        # {'name': 'flow_gpu',
        #  'sampling': 'flow_gpu_all',
        #  'scanner_settings': {
        #      'task_size': 128,
        #      'work_item_size': 8,
        #      'gpu_pool': '3G',
        #      'pipeline_instances_per_node': 1
        #  },
        #  'peak_settings': {
        #      'decoders': 1,
        #      'evaluators': 1,
        #      'tt': '-ss 00:00:00 -t 00:06:00',
        #      'seg': '5',
        #  }},
        {'name': 'caffe',
         'sampling': 'caffe_all',
         'scanner_settings': {
             'io_packet_size': 960,
             'work_packet_size': 96,
             'gpu_pool': '6G',
             'pipeline_instances_per_node': 1,
             'tasks_in_queue_per_pu': 2,
         },
         'peak_settings': {
             'decoders': 1,
             'evaluators': 1,
             'tt': '',
             'seg': '180',
         }},
        # {'name': 'histogram_cpu',
        #  'sampling': 'hist_cpu_all',
        #  'scanner_settings': {
        #      'task_size': 2048,
        #      'work_item_size': 128,
        #      'cpu_pool': '32G',
        #      'pipeline_instances_per_node': 16
        #  },
        #  'peak_settings': {
        #      'decoders': 16,
        #      'evaluators': 16,
        #      'tt': '',
        #      'seg': '180',
        #  }},
        # {'name': 'histogram_gpu',
        #  'sampling': 'hist_cpu_all',
        #  'scanner_settings': {
        #      'task_size': 4096,
        #      'work_item_size': 256,
        #      'gpu_pool': '2G',
        #      'pipeline_instances_per_node': 1
        #  },
        #  'peak_settings': {
        #      'decoders': 1,
        #      'evaluators': 1,
        #      'tt': '',
        #      'seg': '180',
        #  }},
        # {'name': 'flow_cpu',
        #  'sampling': 'flow_cpu_all',
        #  'scanner_settings': {
        #      'task_size': 66,
        #      'work_item_size': 4,
        #      'cpu_pool': '16g',
        #      'pipeline_instances_per_node': 32
        #  },
        #  'peak_settings': {
        #      'decoders': 1,
        #      'evaluators': 32,
        #      'tt': '-ss 00:00:00 -t 00:06:00',
        #      'seg': '5',
        #  }},
    ]

    video_name = 'large'
    results = run_full_comparison('micro', VIDEOS[video_name], tests, True)
    pprint(results)

    ops = ['histogram_cpu',
           'flow_cpu',
           'histogram_gpu',
           'flow_gpu',
           'caffe']
    labels = ['HISTCPU',
              'FLOWCPU',
              'HISTGPU',
              'FLOWGPU',
              'DNN']
    ticks = [1, 5, 10, 15]
    graph.comparison_graphs('micro_{:s}'.format(video_name),
                            VIDEOS[video_name],
                            ops, labels,
                            results['baseline'],
                            results['scanner'],
                            results['peak'],
                            ticks=ticks)

def multi_gpu_benchmarks():
    gpus = CRISSY_NUM_GPUS
    tests = [
        {'name': 'decode_gpu',
         'sampling': 'all',
         'scanner_settings': {
             'io_packet_size': 8448,
             'gpu_pool': '2G',
             'pipeline_instances_per_node': gpus
         }},
        {'name': 'stride_gpu',
         'sampling': 'strided_short',
         'scanner_settings': {
             'io_packet_size': 512,
             'gpu_pool': '2G',
             'pipeline_instances_per_node': gpus
         }},
        {'name': 'gather_gpu',
         'sampling': 'strided_long',
         'scanner_settings': {
             'gpu_pool': '4G',
             'io_packet_size': 1,
             'pipeline_instances_per_node': gpus
         }},
        {'name': 'range_gpu',
         'sampling': 'range',
         'scanner_settings': {
             'task_size': 1280,
             'gpu_pool': '2g',
             'pipeline_instances_per_node': gpus
         }},
        {'name': 'keyframe_gpu',
         'sampling': 'keyframe',
         'scanner_settings': {
             'work_packet_size': 1,
             'io_packet_size': 1,
             'gpu_pool': '2G',
             'pipeline_instances_per_node': gpus
         }},
        {'name': 'histogram_gpu',
         'sampling': 'hist_cpu_all',
         'scanner_settings': {
             'io_packet_size': 4096,
             'work_packet_size': 256,
             'gpu_pool': '2G',
             'pipeline_instances_per_node': gpus
         }},
        {'name': 'flow_gpu',
         'sampling': 'flow_gpu_all',
         'scanner_settings': {
             'io_packet_size': 320,
             'work_packet_size': 32,
             'gpu_pool': '2G',
             #'cpu_pool': 'p32G',
             'pipeline_instances_per_node': gpus
         }},
        {'name': 'caffe',
         'sampling': 'caffe_all',
         'scanner_settings': {
             'io_packet_size': 384,
             'work_packet_size': 96,
             'gpu_pool': '4425M',
             'pipeline_instances_per_node': 1,
             'nodes': ['ocean.pdl.local.cmu.edu:{:d}'.format(p)
                       for p in range(5005, 5010)],
             'tasks_in_queue_per_pu': 2
         }},
    ]
    # decode
    # stride
    # gather
    # range
    # join
    only_graph = True

    video_name = 'large'
    video = VIDEOS[video_name]
    multi_gpu_results = read_results(
        'multi_gpu_results.json')
    #num_gpus = [(1, 1), (1, 2), (1, 4), (2, 4)]
    num_gpus = [(2, 4)]
    hostnames = ['crissy.pdl.local.cmu.edu', 'ocean.pdl.local.cmu.edu']
    multi_gpu_tests = []
    for nodes, gpus in num_gpus:
        num_g = nodes * gpus
        hosts = hostnames[:nodes]
        master = hosts[0]
        for test in tests:
            new_test = copy.deepcopy(test)
            new_test['name'] += '_{:d}'.format(num_g)
            settings = new_test['scanner_settings']
            if 'nodes' in settings:
                settings['nodes'] = [master + ':5001']
                settings['nodes'] += ['{:s}:500{:d}' .format(host, p)
                                      for host in hosts
                                      for p in range(2, 2 + gpus)]
            else:
                settings['pipeline_instances_per_node'] = gpus
                if nodes > 1:
                    settings['nodes'] = [master + ':5001']
                    settings['nodes'] += ['{:s}:5002'.format(host)
                                          for host in hosts]

            if 'nodes' in settings: print(settings['nodes'])
            multi_gpu_tests.append(new_test)
    if not only_graph:
        results = scanner_benchmark(video, multi_gpu_tests)
        for k, v in results.iteritems():
            multi_gpu_results[k] = v
        pprint(results)
        write_results('multi_gpu_results.json', multi_gpu_results)
    else:
        #num_gpus = [(1, 1), (1, 2), (1, 4), (2, 4)]
        num_gpus = [(1, 1), (1, 2), (1, 4)]

    num_gpus = [n * g for n, g in num_gpus]

    # ops = ['decode_gpu',
    #        'stride_gpu',
    #        'gather_gpu',
    #        'range_gpu',
    #        'histogram_gpu',
    #        'caffe',
    #        'flow_gpu']
    # labels = ['DECODE',
    #           'STRIDE',
    #           'GATHER',
    #           'RANGE',
    #           'HIST',
    #           'DNN',
    #           'FLOW']
    ops = ['histogram_gpu',
           'flow_gpu',
           'caffe',]
    labels = ['HIST',
              'FLOW',
              'DNN',]
    ticks = [0, 1, 2, 3, 4]
    graph.multi_gpu_comparison_graphs(video_name,
                                      VIDEOS[video_name],
                                      num_gpus,
                                      ops,
                                      labels,
                                      multi_gpu_results,
                                      labels_on=False)


BENCHMARKS = {
    'load': data_loading_benchmarks,
    'micro': micro_apps_benchmarks,
    'multi': multi_gpu_benchmarks,
    'enc': video_encoding_benchmark_2,
    'surround360': surround360_single_node_benchmark,
    'decode_sol': decode_sol,
    'cpm': cpm_ablation_benchmark
}

def bench_main(args):
    global DEBUG
    DEBUG = args.debug
    test = args.test
    out_dir = args.output_directory
    assert test in BENCHMARKS
    fn = BENCHMARKS[test]
    fn()
