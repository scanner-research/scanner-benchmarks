import subprocess

workloads = ['hist', 'pose']

datasets = ['tvnews']
cpu_nodes = [10, 50, 100, 150, 200, 250, 300]

# Shutdown nodes
subprocess.check_call('cinematography_shutdown_nodes.sh 2 {:d}'.format(500))

cpu_results = []
for nodes in cpu_nodes:
    # Spawn nodes
    subprocess.check_call('cinematography_spawn_nodes.sh {:d}'.format(nodes))
    # Run job
    for workload in workloads:
        for dataset in datasets:
            print('Running: {:s} {:s} {:d}'.format(dataset, workload, nodes))
            subprocess.check_call('python scale_job.py {:s} {:s} {:d}'.format(
                dataset,
                workload,
                nodes))
            # Read total time
            with open('time.txt', 'r') as f:
                total_time = float(f.read())
            print('Total time: {:.2f}'.format(total_time))
    # Shutdown nodes
    subprocess.check_call('cinematography_shutdown_nodes.sh 2 {:d}'.format(nodes))
