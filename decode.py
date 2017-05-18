import cv2
import skimage.measure as measures
from scannerpy import Database, Job, DeviceType
from timeit import default_timer as now

def ssim(A, B):
    return measures.compare_ssim(A, B, multichannel = True)

def decode(db, t, image=False, device=DeviceType.CPU):
    frame = t.as_op().range(0, 10000)
    if image:
        if device == DeviceType.CPU:
            image_type = db.protobufs.ImageDecoderArgs.ANY
        else:
            image_type = db.protobufs.ImageDecoderArgs.JPEG
        frame = db.ops.ImageDecoder(img = frame, image_type = image_type)
    dummy = db.ops.DiscardFrame(ignore = frame, device = device if not image else DeviceType.CPU)
    job = Job(columns = [dummy], name = 'example_dummy')
    start = now()
    out = db.run(job, force = True, work_item_size = 100, pipeline_instances_per_node = 1)
    out.profiler().write_trace('{}.trace'.format(t.name()))
    return now() - start

def main():
    with Database() as db:
        if not db.has_table('example'):
            print 'Ingesting video'
            db.ingest_videos([('example', '/bigdata/wcrichto/videos/movies/fightClub.mp4')], force=True)

        if not db.has_table('example_jpg'):
            print 'Ingesting images'
            num_rows = db.table('example').num_rows()
            rows = [[open('frames-max/{:06d}.jpg'.format(i+1)).read()] for i in range(num_rows)]
            db.new_table('example_jpg', ['jpg'], rows, force=True)

        t = decode(db, db.table('example'))
        print 'Video (CPU)', t

        t = decode(db, db.table('example'), device = DeviceType.GPU)
        print 'Video (GPU)', t

        t = decode(db, db.table('example_jpg'), image=True)
        print 'Images (CPU)', t

        t = decode(db, db.table('example_jpg'), image=True, device = DeviceType.GPU)
        print 'Images (GPU)', t


if __name__ == "__main__":
    main()
