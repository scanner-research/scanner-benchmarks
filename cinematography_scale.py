from scannerpy import Database, DeviceType, Job, BulkJob
from scannerpy.stdlib import parsers
from scipy.spatial import distance
from subprocess import check_call as run
import numpy as np
import cv2
import math
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
import util
import time

try:
    import plotly.offline as offline
    import plotly.graph_objs as go
except ImportError:
    print('You need to install plotly to run this. Try running:\npip install plotly')
    exit()

WINDOW_SIZE = 500

def compute_shot_boundaries(hists):
    # Compute the mean difference between each pair of adjacent frames
    diffs = np.array([np.mean([distance.chebyshev(hists[i-1][j], hists[i][j])
                               for j in range(3)])
                      for i in range(1, len(hists))])
    diffs = np.insert(diffs, 0, 0)
    n = len(diffs)

    # Plot the differences. Look at histogram-diffs.html
    #data = [go.Scatter(x=range(n),y=diffs)]
    #offline.plot(data, filename='histogram-diffs.html')

    # Do simple outlier detection to find boundaries between shots
    boundaries = []
    for i in range(1, n):
        window = diffs[max(i-WINDOW_SIZE,0):min(i+WINDOW_SIZE,n)]
        if diffs[i] - np.mean(window) > 3 * np.std(window):
            boundaries.append(i)
    return boundaries

def main(movie_path):
    total_start = time.time()

    print('Detecting shots in movie {}'.format(movie_path))
    movie_name = os.path.basename(movie_path)
    with open('cinematography_videos.txt', 'r') as f:
        movie_paths = [l.strip() for l in f.readlines()]
    movie_names = [os.path.basename(p) for p in movie_paths]

    # Use GPU kernels if we have a GPU
    with Database() as db:
        print('Loading movie into Scanner database...')
        s = time.time()

        if db.has_gpu():
            device = DeviceType.GPU
        else:
            device = DeviceType.CPU

        ############ ############ ############ ############
        # 0. Ingest the video into the database
        ############ ############ ############ ############
        if not db.table(movie_names[0]):
            movie_tables, failures = db.ingest_videos(
                zip(movie_names, movie_paths), force=True)
        print('Time: {:.1f}s'.format(time.time() - s))
        print('Number of frames in movie: {:d}'.format(movie_table.num_rows()))
        print('Failed videos: {:d}'.format(failures))

        s = time.time()
        ############ ############ ############ ############
        # 1. Run Histogram over the entire video in Scanner
        ############ ############ ############ ############
        print('Computing a color histogram for each frame...')
        frame = db.ops.FrameInput()
        histogram = db.ops.Histogram(
            frame = frame,
            device = device)
        output = db.ops.Output(columns=[histogram])
        jobs = []
        for name in movie_names:
            job = Job(op_args={
                frame: db.table(name).column('frame'),
                output: name + '_hist'
            })
            jobs.append(job)
        bulk_job = BulkJob(output=output, jobs=jobs)
        [hists_table] = db.run(bulk_job, force=True)
        print('\nTime: {:.1f}s, {:.1f} fps'.format(
            time.time() - s,
            movie_table.num_rows() / (time.time() - s)))

        s = time.time()

        exit(0)
        ############ ############ ############ ############
        # 2. Load histograms and compute shot boundaries
        #    in python
        ############ ############ ############ ############
        print('Computing shot boundaries...')
        # Read histograms from disk
        hists = [h for _, h in hists_table.load(['histogram'],
                                                parsers.histograms)]
        boundaries = compute_shot_boundaries(hists)
        print('Found {:d} shots.'.format(len(boundaries)))
        print('Time: {:.1f}s'.format(time.time() - s))

        s = time.time()
        ############ ############ ############ ############
        # 3. Create montage in Scanner
        ############ ############ ############ ############
        print('Creating shot montage...')

        row_length = 16
        rows_per_item = 1
        target_width = 256

        # Compute partial row montages that we will stack together
        # at the end
        frame = db.ops.FrameInput()
        gather_frame = frame.sample()
        sliced_frame = gather_frame.slice()
        montage = db.ops.Montage(
            frame = sliced_frame,
            num_frames = row_length * rows_per_item,
            target_width = target_width,
            frames_per_row = row_length,
            device = device)
        sampled_montage = montage.sample()
        output = db.ops.Output(
            columns=[sampled_montage.unslice().lossless()])

        item_size = row_length * rows_per_item

        starts_remainder = len(boundaries) % item_size
        evenly_divisible = (starts_remainder == 0)
        if not evenly_divisible:
            boundaries = boundaries[0:len(boundaries) - starts_remainder]

        job = Job(op_args={
            frame: movie_table.column('frame'),
            gather_frame: db.sampler.gather(boundaries),
            sliced_frame: db.partitioner.all(item_size),
            sampled_montage: [db.sampler.gather([item_size - 1])
                              for _ in range(len(boundaries) / item_size)],
            output: 'montage_image'
        })
        bulk_job = BulkJob(output=output, jobs=[job])

        [montage_table] = db.run(bulk_job, force=True)

        # Stack all partial montages together
        montage_img = np.zeros((1, target_width * row_length, 3), dtype=np.uint8)
        for idx, img in montage_table.column('montage').load():
            img = np.flip(img, 2)
            montage_img = np.vstack((montage_img, img))

        print('')
        print('Time: {:.1f}s'.format(time.time() - s))

        ############ ############ ############ ############
        # 4. Write montage to disk
        ############ ############ ############ ############
        cv2.imwrite('shots.jpg', montage_img)
        print('Successfully generated shots.jpg')
        print('Total time: {:.2f} s'.format(time.time() - total_start))

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print('Usage: main.py <video_file>')
        exit(1)
    main(sys.argv[1])
