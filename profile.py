from scannerpy import Database, DeviceType, ScannerException
from timeit import default_timer as now
import os

def clear_filesystem_cache():
    os.system('sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"')


def run_trial(db, tasks, op, config):
    print('Running trial...')

    clear_filesystem_cache()
    start = now()
    success = True
    try:
        [output_table] = db.run(tasks, op, force=True, **config)
        prof = output_table.profiler()
        t = now() - start
        print('Trial succeeded, took {:.3f}s'.format(t))
    except ScannerException:
        t = now() - start
        success = False
        prof = None
        print('Trial FAILED after {:.3f}s'.format(t))

    return success, t, prof


def histogram(db, ty=DeviceType.CPU):
    return db.ops.Histogram(device=ty)


def multi_gpu_benchmark():
    db = Database()

    video = '/bigdata/wcrichto/videos/movies/meanGirls.mp4'
    num_gpus = [1, 2, 4]
    pipelines = [
        ('histogram', histogram(db, DeviceType.GPU))]

    [input_table], _ = db.ingest_videos([('test', video)], force=True)
    tasks = db.sampler().all([(input_table.name(), 'test_out')])

    for gpu_count in num_gpus:
        for (name, op) in pipelines:
            s, t, p = run_trial(db, tasks, op, {
                'pipeline_instances_per_node': gpu_count
            })


if __name__ == "__main__":
    multi_gpu_benchmark()
