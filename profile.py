#!/usr/bin/env python

from __future__ import print_function
import argparse
from benchmark import bench_main
from graph import graph_main

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Perform profiling tasks')
    subp = p.add_subparsers(help='sub-command help')
    # Bench
    bench_p = subp.add_parser('bench', help='Run benchmarks')
    bench_p.add_argument('test', type=str,
                         help='Which benchmark to run')
    bench_p.add_argument('output_directory', type=str,
                         help='Where to output results')
    bench_p.add_argument('--debug', action='store_true',
                         help='Enable DB caching while debugging tests')
    bench_p.set_defaults(func=bench_main,
                         test='all',
                         output_directory='benchmark_results')

    # Graphs
    graphs_p = subp.add_parser('graphs', help='Generate graphs from bench')
    graphs_p.set_defaults(func=graph_main)

    args = p.parse_args()
    args.func(args)
