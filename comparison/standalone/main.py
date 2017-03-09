import argparse
import cv2
import numpy as np
import caffe
import toml
from timeit import default_timer as now

NET_PATH = "features/googlenet.toml"
BINS = 16


def output_path(idx):
    return '/tmp/standalone_outputs/videos{}.bin'.format(idx);


def vid_map(paths, f):
    for path_idx, path in enumerate(paths):
        vid = cv2.VideoCapture(path)
        out = open(output_path(path_idx), 'w')
        while (vid.isOpened()):
            _, frame = vid.read()
            if frame is None: break
            out.write(f(frame))
        out.close()


def run_histogram_cpu(paths):
    def hist(frame):
        output = np.zeros((BINS*3))
        for i in range(3):
            cv2.calcHist([frame], [i], None, [BINS], [0,256],
                         output[i*BINS:(i+1)*BINS])
        return output.tobytes()
    vid_map(paths, hist)


last_frame = None
def run_flow_cpu(paths):
    def flow(frame):
        global last_frame
        if last_frame is None: return ''
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
             last_frame, frame,
             0.5, 3, 15, 3, 5, 1.2, 0)
        last_frame = frame
        return ''
    vid_map(paths, flow)


def init_net():
    with open(NET_PATH, 'r') as f:
        net_config = toml.loads(f.read())

    output_layer = net_config['net']['output_layers'][0]

    caffe.set_mode_gpu()
    net = caffe.Net(
        str(net_config['net']['model']),
        str(net_config['net']['weights']),
        caffe.TEST)

    def process(imgs):
        if 'input_width' in net_config['net']:
            width = net_config['net']['input_width']
            height = net_config['net']['input_height']
        else:
            width = img.shape(1)
            height = img.shape(0)
        start = now()
        net.blobs['data'].reshape(*(len(imgs), 3, height, width))
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_channel_swap('data', (2, 1, 0))
        data = np.asarray([transformer.preprocess('data', img) for img in imgs])

        start = now()
        net.forward_all(data=data)

        return net.blobs[output_layer].data

    return process


def run_caffe(paths):
    process = init_net()
    def run(frame):
        return process([frame]).tobytes()
    vid_map(paths, run)


FUNCTIONS = {
    'histogram_cpu': run_histogram_cpu,
    'flow_cpu': run_flow_cpu,
    'caffe': run_caffe
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_type')
    p.add_argument('--paths_file')
    p.add_argument('--operation')

    args = p.parse_args()
    with open(args.paths_file) as f:
        paths = [s.strip() for s in f.readlines()]
    FUNCTIONS[args.operation](paths)

if __name__ == "__main__":
    main()
