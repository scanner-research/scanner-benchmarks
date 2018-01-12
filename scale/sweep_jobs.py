import subprocess
import os

workloads = ['hist', 'pose']

datasets = ['tvnews']
cpu_nodes = [10, 50, 100, 150, 200, 250, 300]

def run_quiet(cmd):
    with open(os.devnull, 'w') as devnull:
        return subprocess.check_call(cmd, shell=True,
                                     stdout=devnull, stderr=devnull)
# Shutdown nodes
print('Running initial node shutdown...')
run_quiet('bash cinematography_shutdown_nodes.sh 2 {:d}'.format(500))

cpu_results = []
for nodes in cpu_nodes:
    print('Spawning {:d} nodes...'.format(nodes))
    # Spawn nodes
    run_quiet('bash cinematography_spawn_nodes.sh {:d}'.format(nodes))
    # Run job
    for workload in workloads:
        for dataset in datasets:
            print('Running: {:s} {:s} {:d}'.format(dataset, workload, nodes))
            rv = run_quiet('python scale_job.py {:s} {:s} {:d}'.format(
                dataset,
                workload,
                nodes))
            print('Return value: {:d}'.format(rv))
            # Read total time
            with open('time.txt', 'r') as f:
                total_time = float(f.read())
            print('Total time: {:.2f}'.format(total_time))
    print('Shutting down {:d} nodes...'.format(nodes))
    # Shutdown nodes
    run_quiet('bash cinematography_shutdown_nodes.sh 2 {:d}'.format(nodes))
