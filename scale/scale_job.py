from scannerpy import Database, DeviceType, Job, BulkJob
from scannerpy.stdlib import parsers
from scipy.spatial import distance
from subprocess import check_call as run
import scannerpy.stdlib.pipelines
import numpy as np
import cv2
import math
import sys
import os.path
import time
import random
import numpy.random


def hist_job(db, device, opts, num_frames, video_names, sampling):
    print('Computing a color histogram for each frame...')
    s = time.time()
    batch = 10000
    for bi, i in enumerate(range(0, len(video_names), batch)):
        print('Batch {:d}...'.format(bi))
        frame = db.ops.FrameInput()
        histogram = db.ops.Histogram(
            frame = frame,
            device = device)
        hist_sample = histogram.sample()
        output = db.ops.Output(columns=[hist_sample])

        jobs = []
        for name, sa in zip(video_names[i:i+batch],
                            sampling[i:i+batch]):
            job = Job(op_args={
                frame: db.table(name).column('frame'),
                hist_sample: sa,
                output: name + '_hist'
            })
            jobs.append(job)
        bulk_job = BulkJob(output=output, jobs=jobs)
        hist_tables = db.run(bulk_job, force=True, **opts)

    print('\nTime: {:.1f}s, {:.1f} fps'.format(
        time.time() - s,
        num_frames / (time.time() - s)))


def openpose_job(db, device, pi, num_frames, video_names, sampling):
    print('Computing pose for each frame...')
    s = time.time()
    poses = scannerpy.stdlib.pipelines.detect_poses(
        db,
        [db.table(tn).column('frame') for tn in video_names],
        sampling,
        'test_poses')
    print('Time: {:.1f}s, {:.1f} fps'.format(
        time.time() - s,
        num_frames / (time.time() - s)))


def main(dataset, workload, num_workers):
    total_start = time.time()

    if dataset == 'tvnews':
        inplace = True
        videos_path = 'tvnews_videos.txt'
    elif dataset == 'cinema':
        inplace = False
        videos_path = 'cinematography_videos.txt'
    elif dataset == 'single':
        inplace = False
        videos_path = 'single_videos.txt'

    if workload == 'hist':
        work_fn = hist_job
        stride = 1
    elif workload == 'pose':
        work_fn = openpose_job
        stride = 30

    with open(videos_path, 'r') as f:
        movie_paths = [l.strip() for l in f.readlines()]
    movie_names = [os.path.basename(p) for p in movie_paths]

    # Use GPU kernels if we have a GPU
    master = 'scanner-apoms-1:5001'
    workers = ['scanner-apoms-{:d}:5002'.format(
        d) for d in range(1, num_workers + 1)]
    with Database(master=master, workers=workers) as db:
        print('Loading movie into Scanner database...')
        s = time.time()

        if db.has_gpu():
            device = DeviceType.GPU
            opts = {
                'pipeline_instances_per_node': 1,
                'work_packet_size': 128,
                'io_packet_size': 4096,
            }
        else:
            device = DeviceType.CPU
            opts = {
                'pipeline_instances_per_node': 8,
                'work_packet_size': 128,
                'io_packet_size': 2048,
            }

        ############ ############ ############ ############
        # 0. Ingest the video into the database
        ############ ############ ############ ############
        if not db.has_table(movie_names[-1]):
            print('Ingest start...')
            batch = 10000
            for i in range(0, len(movie_names), batch):
                print('Ingesting {:d}/{:d}....'.format(i, len(movie_names)))
                has_not = []
                for tn, tp in zip(movie_names[i:i + batch],
                                  movie_paths[i:i+batch]):
                    if not db.has_table(tn) or not db.table(tn).committed():
                        has_not.append((tn, tp))
                if len(has_not) > 0:
                    movie_tables, failures = db.ingest_videos(
                        has_not,
                        force=True,
                        inplace=inplace)
        print('Time: {:.1f}s'.format(time.time() - s))

        num_frames = 0
        failures = 0
        valid_names = []
        for n in movie_names:
            if db.has_table(n):
                num_frames += db.table(n).num_rows()
                valid_names.append(n)
            else:
                failures += 1
        print('Number of frames in movies: {:d}'.format(num_frames))
        print('Failed videos: {:d}'.format(failures))

        if dataset == 'tvnews':
            # Random shots
            sampling = []
            random.seed(1234)
            numpy.random.seed(1234)
            shot_interval_mean = 600
            shot_interval_stddev = 150
            total_frames = 0
            for n in valid_names:
                vid_frames = db.table(n).num_rows()
                shot_interval = numpy.random.normal(loc=shot_interval_mean,
                                                    scale=shot_interval_stddev)
                if vid_frames < shot_interval + 1:
                    start = 0
                    end = vid_frames
                else:
                    start = random.randint(0, vid_frames - shot_interval)
                    end = start + shot_interval
                sampling.append(db.sampler.strided_range(start, end, stride))
                total_frames += (end - start)
            num_frames = total_frames
            print('Number of frames in movies: {:d}'.format(num_frames))
        elif dataset == 'cinema':
            # Everything
            sampling = []
            for n in valid_names:
                sampling.append(db.sampler.strided(stride))
        elif dataset == 'single':
            sampling = []
            for n in valid_names:
                sampling.append(db.sampler.strided(stride))

        s = time.time()
        ############ ############ ############ ############
        # 1. Run Histogram over the entire video in Scanner
        ############ ############ ############ ############

        work_fn(db, device, opts, num_frames, valid_names, sampling)

        exit(0)

if __name__ == "__main__":
    dataset = sys.argv[1]
    workload = sys.argv[2]
    num_workers = int(sys.argv[3])
    main(dataset, workload, num_workers)
