import subprocess
import os

iterations = 3

run_cpu_tests = True
run_gpu_tests = False

#workloads = ['hist', 'pose']
cpu_workloads = ['hist']
gpu_workloads = ['hist', 'pose']

datasets = ['tvnews']

num_gpus_per_node = 2

cpu_nodes = [10, 50, 100, 150, 200, 250, 300]
gpu_nodes = [10, 25, 50, 75, 100, 128]

def run_quiet(cmd):
    with open(os.devnull, 'w') as devnull:
        return subprocess.call(cmd, shell=True,
                               stdout=devnull, stderr=devnull)
class RetryException(Exception):
    pass

ATTEMPTS = 3

# Shutdown nodes
if run_cpu_tests:
    print('Running cpu tests')
    print('Running initial node shutdown...')
    run_quiet('bash cinematography_shutdown_nodes.sh 2 {:d}'.format(500))
    cpu_results = []
    for i in range(iterations):
        print('Iteration {:d}'.format(i))
        for nodes in cpu_nodes:
            attempt = 0
            while attempt < ATTEMPTS:
                attempt += 1
                try:
                    print('Spawning {:d} nodes...'.format(nodes))
                    # Spawn nodes
                    run_quiet('bash cinematography_spawn_nodes.sh {:d}'.format(nodes))
                    # Run job
                    for workload in cpu_workloads:
                        for dataset in datasets:
                            print('Running: {:s} {:s} {:d}'.format(dataset, workload, nodes))
                            rv = run_quiet('python scale_job.py {:s} {:s} {:d}'.format(
                                dataset,
                                workload,
                                nodes))
                            if rv != 0:
                                print('Return value was: {:d}, retrying'.format(rv))
                                raise RetryException()
                            # Read total time
                            with open('time.txt', 'r') as f:
                                total_time = float(f.read())
                            print('Total time: {:.2f}'.format(total_time))
                    break
                except RetryException:
                    pass
                finally:
                    print('Shutting down {:d} nodes...'.format(nodes))
                    # Shutdown nodes
                    run_quiet('bash cinematography_shutdown_nodes.sh 2 {:d}'.format(nodes))


if run_gpu_tests:
    gpu_results = []
    print('Running gpu tests')
    print('Running initial node shutdown...')
    run_quiet('bash cinematography_shutdown_nodes_gpu.sh 2 {:d}'.format(500))
    for i in range(iterations):
        print('Iteration {:d}'.format(i))
        for nodes in gpu_nodes:
            while attempt < ATTEMPTS:
                attempt += 1
                try:
                    print('Spawning {:d} nodes...'.format(nodes))
                    # Spawn nodes
                    run_quiet('bash cinematography_spawn_nodes_gpu.sh {:d}'.format(nodes))
                    # Run job
                    for workload in gpu_workloads:
                        for dataset in datasets:
                            print('Running: {:s} {:s} {:d}'.format(dataset, workload, nodes))
                            rv = run_quiet('python scale_job.py {:s} {:s} {:d}'.format(
                                dataset,
                                workload,
                                nodes))
                            if rv != 0:
                                print('Return value was: {:d}, retrying'.format(rv))
                                raise RetryException()
                            # Read total time
                            with open('time.txt', 'r') as f:
                                total_time = float(f.read())
                            print('Total time: {:.2f}'.format(total_time))
                    break
                except RetryException:
                    pass
                finally:
                    print('Shutting down {:d} nodes...'.format(nodes))
                    # Shutdown nodes
                    run_quiet('bash cinematography_shutdown_nodes_gpu.sh 2 {:d}'.format(nodes))
