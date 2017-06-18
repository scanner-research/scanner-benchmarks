/* Copyright 2016 Carnegie Mellon University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "peak_cpu_video_decoder.h"
#include "peak_gpu_video_decoder.h"

// #include "scanner/util/util.h"
// #include "scanner/util/h264.h"

#include "scanner/engine/video_index_entry.h"
#include "scanner/engine/load_worker.h"
#include "scanner/video/decoder_automata.h"
#include "scanner/util/profiler.h"

#include "util/net_descriptor.h"
#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/net.hpp"

#include "generator_genfiles/caffe_input_transformer_gpu/caffe_input_transformer_gpu.h"
#include "HalideRuntimeCuda.h"
#include "Halide.h"
#include "scanner/util/halide_context.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <thread>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/errors.hpp>

#ifdef HAVE_CUDA
#include "scanner/util/cuda.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/error.h"
#include "libavutil/frame.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libswscale/swscale.h"
}

#include <iostream>
#include <fstream>
#include <thread>
#include <unistd.h>
#include <mutex>

using namespace scanner;

namespace po = boost::program_options;
namespace cvc = cv::cuda;

using scanner::Queue;

enum OpType {
  Histogram,
  Flow,
  Caffe
};

struct BufferHandle {
  u8* buffer;
  int elements;
};

struct SaveHandle {
  u8* buffer;
  cudaStream_t stream;
  int elements;
};

using DecoderFn = std::function<void(int, Queue<int64_t>&, Queue<u8*> &,
                                     Queue<BufferHandle> &)>;

using WorkerFn = std::function<void(int, Queue<u8 *> &, Queue<BufferHandle> &,
                                    Queue<SaveHandle> &, Queue<SaveHandle> &)>;

const int NUM_BUFFERS = 3;
int BATCH_SIZE = 128;      // Batch size for network
const int NET_BATCH_SIZE = 128;      // Batch size for network
const int BINS = 16;
const std::string NET_PATH = "nets/googlenet.toml";

int GPUS_PER_NODE = 1;           // GPUs to use per node
int width;
int height;
size_t output_element_size;
size_t output_buffer_size;

std::string PATH;
std::string OPERATION;
std::mutex TIMINGS_MUTEX;
std::map<std::string, double> TIMINGS;

std::string DECODE_TYPE;
std::string DECODE_ARGS;

std::vector<std::string> VIDEO_PATHS;
std::vector<std::tuple<int64_t, int64_t>> TASK_RANGES;

scanner::internal::VideoIndexEntry INDEX_ENTRY;
int64_t SCANNER_TABLE_ID;
int64_t SCANNER_COLUMN_ID;
int64_t STRIDE = 1;

bool IS_CPU;

std::string output_path() {
  i32 idx = 0;
  return "/tmp/peak_outputs/videos" + std::to_string(idx) + ".bin";
}

struct CodecState {
  AVPacket packet;
  AVFrame* frame;
  AVFormatContext* format_context;
  AVCodec* in_codec;
  AVCodecContext* cc;
  SwsContext* sws_context;
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 34, 0)
  AVCodecParameters* in_cc_params;
#endif
  i32 video_stream_index;
  AVBitStreamFilterContext* annexb;
};

bool setup_video_codec(CodecState& state, const std::string& path) {
  LOG(INFO) << "Setting up video codec";
  av_init_packet(&state.packet);
  state.frame = av_frame_alloc();
  state.format_context = avformat_alloc_context();

  avformat_open_input(&state.format_context, path.c_str(), 0, 0);

  // Read file header
  LOG(INFO) << "Opening input file to read format";
  if (avformat_open_input(&state.format_context, NULL, NULL, NULL) < 0) {
    LOG(ERROR) << "open input failed";
    return false;
  }
  // Some formats don't have a header
  LOG(INFO) << "Find stream info";
  if (avformat_find_stream_info(state.format_context, NULL) < 0) {
    LOG(ERROR) << "find stream info failed";
    return false;
  }

  LOG(INFO) << "Dump format";
  av_dump_format(state.format_context, 0, NULL, 0);

  // Find the best video stream in our input video
  LOG(INFO) << "Find best stream";
  state.video_stream_index = av_find_best_stream(
      state.format_context, AVMEDIA_TYPE_VIDEO, -1 /* auto select */,
      -1 /* no related stream */, &state.in_codec, 0 /* flags */);
  if (state.video_stream_index < 0) {
    LOG(ERROR) << "could not find best stream";
    return false;
  }

  AVStream const* const in_stream =
      state.format_context->streams[state.video_stream_index];

  LOG(INFO) << "Find decoder";
  state.in_codec = avcodec_find_decoder(AV_CODEC_ID_H264);
  if (state.in_codec == NULL) {
    LOG(FATAL) << "could not find h264 decoder";
  }

  state.cc = avcodec_alloc_context3(state.in_codec);
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 34, 0)
  state.in_cc_params = avcodec_parameters_alloc();
  if (avcodec_parameters_from_context(state.in_cc_params, in_stream->codec) <
      0) {
    LOG(ERROR) << "could not copy codec params from input stream";
    return false;
  }
  if (avcodec_parameters_to_context(state.cc, state.in_cc_params) < 0) {
    LOG(ERROR) << "could not copy codec params to in cc";
    return false;
  }
#else
  if (avcodec_copy_context(state.cc, in_stream->codec) < 0) {
    LOG(ERROR) << "could not copy codec params to in cc";
    return false;
  }
#endif

  state.cc->thread_count = 16;
  state.cc->refcounted_frames = 1;
  if (avcodec_open2(state.cc, state.in_codec, NULL) < 0) {
    LOG(ERROR) << "could not open codec";
    return false;
  }

//state.annexb = av_bitstream_filter_init("h264_mp4toannexb");

  return true;
}

void cleanup_video_codec(CodecState state) {
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(55, 53, 0)
  avcodec_free_context(&state.cc);
#else
  avcodec_close(state.cc);
  av_freep(&state.cc);
#endif
  avformat_close_input(&state.format_context);
  av_frame_free(&state.frame);
  //av_bitstream_filter_close(state.annexb);
}

void ffmpeg_decoder_worker(int gpu_device_id, Queue<int64_t> &task_ids,
                           Queue<u8 *> &free_buffers,
                           Queue<BufferHandle> &decoded_frames) {
  double decode_time = 0;

  int64_t frame_size = width * height * 4 * sizeof(u8);
  int64_t frame = 0;
  assert(IS_CPU);
  while (true) {
    int64_t task_id;
    task_ids.pop(task_id);

    if (task_id == -1) {
      break;
    }

    std::string path = VIDEO_PATHS[task_id];
    printf("popped %s\n", path.c_str());
    CodecState state;
    bool success = setup_video_codec(state, path);
    LOG_IF(FATAL, !success) << "Did not setup video codec";

    int64_t frame = 0;
    bool video_done = false;
    bool to_receive = false;
    while (!video_done) {
      u8 *buffer;
      free_buffers.pop(buffer);

      int64_t buffer_frame = 0;
      while (buffer_frame < BATCH_SIZE) {
        auto decode_start = scanner::now();
        int error;
        if (!to_receive) {
          error = av_read_frame(state.format_context, &state.packet);
          if (error == AVERROR_EOF) {
            printf("EOF!\n");
            video_done = true;
            break;
          }
          if (state.packet.stream_index != state.video_stream_index) {
            av_packet_unref(&state.packet);
            continue;
          }

          error = avcodec_send_packet(state.cc, &state.packet);
          if (error != AVERROR_EOF) {
            if (error < 0) {
              char err_msg[256];
              av_strerror(error, err_msg, 256);
              fprintf(stderr, "Error while sending packet (%d): %s\n", error,
                      err_msg);
              LOG(FATAL) << "Error while sending packet";
            }
          }
          av_packet_unref(&state.packet);
        }
        while (buffer_frame < BATCH_SIZE) {
          error = avcodec_receive_frame(state.cc, state.frame);
          if (error == AVERROR_EOF) {
            to_receive = false;
            av_frame_unref(state.frame);
            break;
          }
          if (error == 0) {
            to_receive = true;
            u8* scale_buffer = buffer + buffer_frame * frame_size;

            uint8_t *out_slices[4];
            int out_linesizes[4];
            int required_size = av_image_fill_arrays(
                out_slices, out_linesizes, scale_buffer, AV_PIX_FMT_RGB24,
                width, height, 1);

            AVPixelFormat decoder_pixel_format = state.cc->pix_fmt;
            if (state.sws_context == nullptr) {
              printf("sws %d\n", decoder_pixel_format);
              state.sws_context = sws_getContext(
                  width, height, decoder_pixel_format, width,
                  height, AV_PIX_FMT_RGB24, SWS_BICUBIC, NULL, NULL, NULL);
            }
            if (state.sws_context == NULL) {
              LOG(FATAL) << "Could not get sws context";
            }

            av_frame_unref(state.frame);
            if (sws_scale(state.sws_context, state.frame->data,
                          state.frame->linesize, 0, state.frame->height,
                          out_slices, out_linesizes) < 0) {
              LOG(FATAL) << "Sws_scale failed";
              exit(EXIT_FAILURE);
            }

            av_frame_unref(state.frame);
            frame++;
            buffer_frame++;
            continue;
          } else if (error == AVERROR(EAGAIN)) {
            to_receive = false;
            break;
          } else {
            char err_msg[256];
            av_strerror(error, err_msg, 256);
            fprintf(stderr, "Error while receiving frame (%d): %s\n", error,
                    err_msg);
            exit(-1);
          }
        }
      }

      printf("video %s, %d\n", path.c_str(), frame);
      BufferHandle h;
      h.buffer = buffer;
      h.elements = buffer_frame;
      decoded_frames.push(h);
    }
    cleanup_video_codec(state);
  }
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  TIMINGS["decode"] += decode_time;
}

void scanner_cpu_decoder_worker(int gpu_device_id,
                                Queue<int64_t> &task_ids,
                                Queue<u8 *> &free_buffers,
                                Queue<BufferHandle> &decoded_frames) {
  double decode_time = 0;

  std::unique_ptr<storehouse::StorageConfig> config(
      storehouse::StorageConfig::make_posix_config());
  std::unique_ptr<storehouse::StorageBackend> storage(
      storehouse::StorageBackend::make_from_config(config.get()));

  scanner::internal::VideoIndexEntry index_entry =
      scanner::internal::read_video_index(storage.get(), SCANNER_TABLE_ID,
                                          SCANNER_COLUMN_ID, 0);

  int64_t frame_size = width * height * 4 * sizeof(u8);
  int64_t frame = 0;

  scanner::Profiler profiler(scanner::now());
  std::unique_ptr<scanner::internal::DecoderAutomata> decoder(
      new scanner::internal::DecoderAutomata(
          scanner::CPU_DEVICE, 1,
          scanner::internal::VideoDecoderType::SOFTWARE));
  decoder->set_profiler(&profiler);
  while (true) {
    int64_t task_id;
    task_ids.pop(task_id);

    if (task_id == -1) {
      break;
    }

    int64_t task_start, task_end;
    std::tie(task_start, task_end) = TASK_RANGES.at(task_id);

    // Read encoded video
    std::vector<int64_t> rows;
    for (int64_t i = task_start; i < task_end; i += STRIDE) {
      rows.push_back(i);
    }
    scanner::ElementList element_list;
    scanner::internal::read_video_column(profiler, index_entry, rows, 0,
                                         element_list);

    // Feed decode args into decoder
    std::vector<proto::DecodeArgs> decode_args;
    for (Element element : element_list) {
      decode_args.emplace_back();
      scanner::proto::DecodeArgs &da = decode_args.back();
      google::protobuf::io::ArrayInputStream in_stream(element.buffer,
                                                       element.size);
      google::protobuf::io::CodedInputStream cstream(&in_stream);
      cstream.SetTotalBytesLimit(element.size + 1, element.size + 1);
      bool result = da.ParseFromCodedStream(&cstream);
      assert(result);
      delete_element(CPU_DEVICE, element);
    }
    decoder->initialize(decode_args);

    int64_t frame = task_start;
    bool video_done = false;
    while (frame < task_end) {
      u8 *buffer;
      free_buffers.pop(buffer);

      int64_t batch = std::min((int64_t)BATCH_SIZE, (task_end - frame) / STRIDE);
      auto decode_start = scanner::now();
      decoder->get_frames(buffer, batch);
      BufferHandle h;
      h.buffer = buffer;
      h.elements = batch;
      decoded_frames.push(h);

      frame += batch * STRIDE;
    }
  }
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  TIMINGS["decode"] += decode_time;
  TIMINGS["frames_fed"] +=
      profiler.get_counters().at("frames_fed") * 1000000000.0;
  TIMINGS["frames_used"] +=
      profiler.get_counters().at("frames_used") * 1000000000.0;
  TIMINGS["frames_decoded"] +=
      profiler.get_counters().at("frames_decoded") * 1000000000.0;
}

void scanner_gpu_decoder_worker(int gpu_device_id,
                                Queue<int64_t> &task_ids,
                                Queue<u8 *> &free_buffers,
                                Queue<BufferHandle> &decoded_frames) {
  double decode_time = 0;

  std::unique_ptr<storehouse::StorageConfig> config(
      storehouse::StorageConfig::make_posix_config());
  std::unique_ptr<storehouse::StorageBackend> storage(
      storehouse::StorageBackend::make_from_config(config.get()));

  scanner::internal::VideoIndexEntry index_entry =
      scanner::internal::read_video_index(storage.get(), SCANNER_TABLE_ID,
                                          SCANNER_COLUMN_ID, 0);

  int64_t frame_size = width * height * 4 * sizeof(u8);
  int64_t frame = 0;

  scanner::Profiler profiler(scanner::now());
  std::unique_ptr<scanner::internal::DecoderAutomata> decoder(
      new scanner::internal::DecoderAutomata(
          scanner::DeviceHandle{DeviceType::GPU, 0}, 1,
          scanner::internal::VideoDecoderType::NVIDIA));
  decoder->set_profiler(&profiler);
  while (true) {
    int64_t task_id;
    task_ids.pop(task_id);

    if (task_id == -1) {
      break;
    }

    int64_t task_start, task_end;
    std::tie(task_start, task_end) = TASK_RANGES.at(task_id);
    printf("popped %ld, %ld\n", task_start, task_end);

    // Read encoded video
    std::vector<int64_t> rows;
    for (int64_t i = task_start; i < task_end; i += STRIDE) {
      rows.push_back(i);
    }
    scanner::Profiler profiler(scanner::now());
    scanner::ElementList element_list;
    scanner::internal::read_video_column(profiler, index_entry, rows, 0,
                                         element_list);

    // Feed decode args into decoder
    std::vector<proto::DecodeArgs> decode_args;
    for (Element element : element_list) {
      decode_args.emplace_back();
      scanner::proto::DecodeArgs &da = decode_args.back();
      google::protobuf::io::ArrayInputStream in_stream(element.buffer,
                                                       element.size);
      google::protobuf::io::CodedInputStream cstream(&in_stream);
      cstream.SetTotalBytesLimit(element.size + 1, element.size + 1);
      bool result = da.ParseFromCodedStream(&cstream);
      assert(result);
      delete_element(CPU_DEVICE, element);
    }
    decoder->initialize(decode_args);

    int64_t frame = task_start;
    bool video_done = false;
    while (frame < task_end) {
      u8 *buffer;
      free_buffers.pop(buffer);

      int64_t batch = std::min((int64_t)BATCH_SIZE, (task_end - frame) / STRIDE);
      auto decode_start = scanner::now();
      decoder->get_frames(buffer, batch);
      BufferHandle h;
      h.buffer = buffer;
      h.elements = batch;
      decoded_frames.push(h);

      frame += batch * STRIDE;
    }
  }
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  TIMINGS["decode"] += decode_time;
  TIMINGS["frames_fed"] +=
      profiler.get_counters().at("frames_fed") * 1000000000.0;
  TIMINGS["frames_used"] +=
      profiler.get_counters().at("frames_used") * 1000000000.0;
  TIMINGS["frames_decoded"] +=
      profiler.get_counters().at("frames_decoded") * 1000000000.0;
}


void opencv_decoder_worker(int gpu_device_id, Queue<int64_t> &task_ids,
                           Queue<u8 *> &free_buffers,
                           Queue<BufferHandle> &decoded_frames) {
  double decode_time = 0;

  int64_t frame_size = width * height * 4 * sizeof(u8);
  int64_t frame = 0;
  std::vector<u8> frame_buffer(frame_size);

  cv::VideoCapture cap;
  cv::Ptr<cv::cudacodec::VideoReader> video;
  while (true) {
    int64_t task_id;
    task_ids.pop(task_id);

    if (task_id == -1) {
      break;
    }

    std::string path = VIDEO_PATHS.at(task_id);

    printf("popped %s\n", path.c_str());
    if (IS_CPU) {
      cap.open(path);
    } else {
      video = cv::cudacodec::createVideoReader(path);
    }

    int64_t frame = 0;
    bool video_done = false;
    while (!video_done) {
      u8 *buffer;
      free_buffers.pop(buffer);

      int64_t buffer_frame = 0;
      while (buffer_frame < BATCH_SIZE) {
        bool valid;
        auto decode_start = scanner::now();
        if (IS_CPU) {
          cv::Mat cpu_image(width, height, CV_8UC3,
                            buffer + buffer_frame * frame_size);
          valid = cap.read(cpu_image);
        } else {
          cvc::GpuMat gpu_image(width, height, CV_8UC4,
                                buffer + buffer_frame * frame_size);
          valid = video->nextFrame(gpu_image);
        }
        decode_time += scanner::nano_since(decode_start);
        if (!valid) {
          video_done = true;
          break;
        }
        buffer_frame++;
        frame++;
      }

      printf("video %s, %d\n", path.c_str(), frame);
      BufferHandle h;
      h.buffer = buffer;
      h.elements = buffer_frame;
      decoded_frames.push(h);
    }
  }
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  TIMINGS["decode"] += decode_time;
}

void save_worker(int gpu_device_id, Queue<SaveHandle> &save_buffers,
                 Queue<SaveHandle> &free_output_buffers) {
  int64_t frame_size = width * height * 3 * sizeof(u8);
  int64_t frame = 0;
  double save_time = 0;

  u8* buf;
  if (!IS_CPU) {
    cudaMallocHost(&buf, output_buffer_size);
  }

  std::ofstream outfile(output_path(),
                        std::fstream::binary | std::fstream::trunc);
  assert(outfile.good());
  while (true) {
    SaveHandle handle;
    save_buffers.pop(handle);

    if (handle.buffer == nullptr) {
      break;
    }

    // Copy data down
    auto save_start = scanner::now();
    if (IS_CPU) {
      buf = handle.buffer;
    } else {
      cudaMemcpyAsync(buf, handle.buffer, output_element_size * handle.elements,
                      cudaMemcpyDefault, handle.stream);
      // Sync so we know it is done
      cudaStreamSynchronize(handle.stream);
    }
    // Write out
    if (OPERATION != "flow_cpu" &&
        OPERATION != "flow_gpu") {
      outfile.write((char*)buf, output_element_size * handle.elements);
    }
    save_time += scanner::nano_since(save_start);

    free_output_buffers.push(handle);
  }
  auto save_start = scanner::now();
  outfile.close();
  save_time += scanner::nano_since(save_start);
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  TIMINGS["save"] += save_time;
  //cudaFreeHost(buf);
}

void video_decode_cpu_worker(int gpu_device_id, Queue<u8 *> &free_buffers,
                             Queue<BufferHandle> &decoded_frames,
                             Queue<SaveHandle> &free_output_buffers,
                             Queue<SaveHandle> &save_buffers) {
  double setup_time = 0;
  double load_time = 0;
  double histo_time = 0;
  double save_time = 0;

  int frame_size = width * height * 4 * sizeof(u8);

  auto setup_start = scanner::now();

  int64_t frame = 0;
  while (true) {
    BufferHandle buffer_handle;
    decoded_frames.pop(buffer_handle);
    u8* buffer = buffer_handle.buffer;
    int elements = buffer_handle.elements;
    if (buffer == nullptr) {
      break;
    }

    SaveHandle save_handle;
    free_output_buffers.pop(save_handle);
    u8* output_buffer = save_handle.buffer;
    save_handle.elements = elements;

    save_buffers.push(save_handle);
    free_buffers.push(buffer);
  }
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  TIMINGS["setup"] += setup_time;
  TIMINGS["load"] += load_time;
  TIMINGS["eval"] += histo_time;
  TIMINGS["save"] += save_time;
}

void video_decode_worker(int gpu_device_id, Queue<u8 *> &free_buffers,
                         Queue<BufferHandle> &decoded_frames,
                         Queue<SaveHandle> &free_output_buffers,
                         Queue<SaveHandle> &save_buffers) {
  double setup_time = 0;
  double load_time = 0;
  double histo_time = 0;
  double save_time = 0;

  int frame_size = width * height * 4 * sizeof(u8);

  auto setup_start = scanner::now();

  int64_t frame = 0;
  while (true) {
    BufferHandle buffer_handle;
    decoded_frames.pop(buffer_handle);
    u8* buffer = buffer_handle.buffer;
    int elements = buffer_handle.elements;
    if (buffer == nullptr) {
      break;
    }

    SaveHandle save_handle;
    free_output_buffers.pop(save_handle);
    u8* output_buffer = save_handle.buffer;
    save_handle.elements = elements;

    save_buffers.push(save_handle);
    free_buffers.push(buffer);
  }
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  TIMINGS["setup"] += setup_time;
  TIMINGS["load"] += load_time;
  TIMINGS["eval"] += histo_time;
  TIMINGS["save"] += save_time;
}

void video_histogram_cpu_worker(int gpu_device_id,
                                Queue<u8 *> &free_buffers,
                                Queue<BufferHandle> &decoded_frames,
                                Queue<SaveHandle> &free_output_buffers,
                                Queue<SaveHandle> &save_buffers) {
  double setup_time = 0;
  double load_time = 0;
  double histo_time = 0;
  double save_time = 0;

  // Set ourselves to the correct GPU
  int frame_size = width * height * 4 * sizeof(u8);

  auto setup_start = scanner::now();
  std::vector<cv::Mat> planes;
  for (int i = 0; i < 3; ++i) {
    planes.push_back(cv::Mat(height, width, CV_8UC1));
  }

  cv::Mat hist;
  cv::Mat hist_32s;

  int64_t frame = 0;
  while (true) {
    BufferHandle buffer_handle;
    decoded_frames.pop(buffer_handle);
    u8* buffer = buffer_handle.buffer;
    int elements = buffer_handle.elements;
    if (buffer == nullptr) {
      break;
    }

    SaveHandle save_handle;
    free_output_buffers.pop(save_handle);
    u8* output_buffer = save_handle.buffer;
    save_handle.elements = elements;

    for (int i = 0; i < elements; ++i) {
      cv::Mat image(height, width, CV_8UC3, buffer + i * frame_size);
      auto histo_start = scanner::now();
      float range[] = {0, 256};
      const float* histRange = {range};
      u8* output_buf = output_buffer + i * output_element_size;
      for (i32 j = 0; j < 3; ++j) {
        int channels[] = {j};
        cv::Mat out(BINS, 1, CV_32S, output_buf + j * BINS * sizeof(int));
        cv::calcHist(&image, 1, channels, cv::Mat(),
                     out,
                     1, &BINS,
                     &histRange);
      }
      histo_time += scanner::nano_since(histo_start);
    }

    save_buffers.push(save_handle);
    free_buffers.push(buffer);
  }
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  TIMINGS["setup"] += setup_time;
  TIMINGS["load"] += load_time;
  TIMINGS["eval"] += histo_time;
  TIMINGS["save"] += save_time;
}

void video_histogram_gpu_worker(int gpu_device_id, Queue<u8 *> &free_buffers,
                                Queue<BufferHandle> &decoded_frames,
                                Queue<SaveHandle> &free_output_buffers,
                                Queue<SaveHandle> &save_buffers) {
  double setup_time = 0;
  double load_time = 0;
  double histo_time = 0;
  double save_time = 0;

  // Set ourselves to the correct GPU
  cv::cuda::setDevice(gpu_device_id);

  int frame_size = width * height * 4 * sizeof(u8);

  auto setup_start = scanner::now();
  std::vector<cvc::GpuMat> planes;
  for (int i = 0; i < 4; ++i) {
    planes.push_back(cvc::GpuMat(height, width, CV_8UC1));
  }

  cvc::GpuMat hist(1, BINS, CV_32S);
  cvc::GpuMat out_gpu(1, BINS * 3, CV_32S);

  int64_t frame = 0;
  while (true) {
    BufferHandle buffer_handle;
    decoded_frames.pop(buffer_handle);
    u8* buffer = buffer_handle.buffer;
    int elements = buffer_handle.elements;
    if (buffer == nullptr) {
      break;
    }

    SaveHandle save_handle;
    free_output_buffers.pop(save_handle);
    u8* output_buffer = save_handle.buffer;
    save_handle.elements = elements;

    cvc::Stream stream = cvc::StreamAccessor::wrapStream(save_handle.stream);
    for (int i = 0; i < elements; ++i) {
      cvc::GpuMat image(height, width, CV_8UC3, buffer + i * frame_size);
      auto histo_start = scanner::now();
      cvc::split(image, planes, stream);
      for (int j = 0; j < 3; ++j) {
        cvc::histEven(planes[j], hist, BINS, 0, 256, stream);
        hist.copyTo(out_gpu(cv::Rect(j * BINS, 0, BINS, 1)), stream);
      }
      cudaMemcpyAsync(output_buffer + i * output_element_size, out_gpu.data,
                      output_element_size, cudaMemcpyDefault,
                      save_handle.stream);
      histo_time += scanner::nano_since(histo_start);
    }

    save_buffers.push(save_handle);
    free_buffers.push(buffer);
  }
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  TIMINGS["setup"] += setup_time;
  TIMINGS["load"] += load_time;
  TIMINGS["eval"] += histo_time;
  TIMINGS["save"] += save_time;
}

void video_flow_cpu_worker(int gpu_device_id, Queue<u8 *> &free_buffers,
                           Queue<BufferHandle> &decoded_frames,
                           Queue<SaveHandle> &free_output_buffers,
                           Queue<SaveHandle> &save_buffers) {
  double load_time = 0;
  double eval_time = 0;
  double save_time = 0;
  // Set ourselves to the correct GPU
  cv::cuda::setDevice(gpu_device_id);

  int frame_size = width * height * 4 * sizeof(u8);

  std::vector<cv::Mat> gray;
  for (int i = 0; i < 2; ++i) {
    gray.emplace_back(height, width, CV_8UC1);
  }

  auto flow = cv::FarnebackOpticalFlow::create(3, 0.5, false, 15, 3, 5, 1.2, 0);

  bool first = true;
  int64_t frame = 0;
  int flow_invocations = 0;
  while (true) {
    BufferHandle buffer_handle;
    decoded_frames.pop(buffer_handle);
    u8* buffer = buffer_handle.buffer;
    int elements = buffer_handle.elements;
    if (buffer == nullptr) {
      break;
    }

    SaveHandle save_handle;
    free_output_buffers.pop(save_handle);
    save_handle.elements = elements;

    u8* output_buffer = save_handle.buffer;

    // Load the first frame
    auto eval_first = scanner::now();
    int i = 0;
    if (first) {
      cv::Mat image(height, width, CV_8UC3, buffer);
      cv::cvtColor(image, gray[0], CV_BGR2GRAY);
      i += 1;
      first = false;
    }
    eval_time += scanner::nano_since(eval_first);
    bool done = false;
    for (; i < elements; ++i) {
      int curr_idx = (i + 1) % 2;
      int prev_idx = (i) % 2;
      cv::Mat output_flow(height, width, CV_32FC2,
                          output_buffer + i * output_element_size);
      cv::Mat image(height, width, CV_8UC3, buffer + i * frame_size);

      auto eval_start = scanner::now();
      cv::cvtColor(image, gray[curr_idx], CV_BGR2GRAY);
      flow->calc(gray[prev_idx], gray[curr_idx], output_flow);
      flow_invocations++;
      eval_time += scanner::nano_since(eval_start);
    }

    save_buffers.push(save_handle);
    free_buffers.push(buffer);
  }
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  printf("flow invocs %d\n", flow_invocations);
  TIMINGS["load"] += load_time;
  TIMINGS["eval"] += eval_time;
  TIMINGS["save"] += save_time;
}

void video_flow_gpu_worker(int gpu_device_id, Queue<u8 *> &free_buffers,
                           Queue<BufferHandle> &decoded_frames,
                           Queue<SaveHandle> &free_output_buffers,
                           Queue<SaveHandle> &save_buffers) {
  double load_time = 0;
  double eval_time = 0;
  double save_time = 0;
  // Set ourselves to the correct GPU
  cv::cuda::setDevice(gpu_device_id);
  cv::cuda::setBufferPoolUsage(true);
  cv::cuda::setBufferPoolConfig(gpu_device_id, 50 * 1024 * 1024, 3);

  int frame_size = width * height * 4 * sizeof(u8);

  std::vector<cvc::GpuMat> gray;
  for (int i = 0; i < 2; ++i) {
    gray.emplace_back(height, width, CV_8UC1);
  }

  cv::Ptr<cvc::DenseOpticalFlow> flow =
      cvc::FarnebackOpticalFlow::create(3, 0.5, false, 15, 3, 5, 1.2, 0);

  std::vector<cvc::Stream> streams;
  int64_t frame = 0;
  while (true) {
    BufferHandle buffer_handle;
    decoded_frames.pop(buffer_handle);
    u8* buffer = buffer_handle.buffer;
    int elements = buffer_handle.elements;
    if (buffer == nullptr) {
      break;
    }

    SaveHandle save_handle;
    free_output_buffers.pop(save_handle);
    save_handle.elements = elements;

    u8* output_buffer = save_handle.buffer;
    cvc::Stream stream = cvc::StreamAccessor::wrapStream(save_handle.stream);

    // Load the first frame
    auto eval_first = scanner::now();
    gray.resize(elements);
    streams.resize(elements);
    for (int i = 0; i < elements; ++i) {
      cvc::GpuMat image(height, width, CV_8UC4, buffer + i * frame_size);
      cvc::cvtColor(image, gray[i], CV_BGRA2GRAY, 0, streams[i]);
    }
    eval_time += scanner::nano_since(eval_first);
    bool done = false;
    for (int i = 1; i < elements; ++i) {
      int curr_idx = i;
      int prev_idx = (i - 1);
      cvc::GpuMat output_flow_gpu(height, width, CV_32FC2,
                                  output_buffer + i * output_element_size);

      auto eval_start = scanner::now();
      flow->calc(gray[prev_idx], gray[curr_idx], output_flow_gpu, stream);
      eval_time += scanner::nano_since(eval_start);
    }

    save_buffers.push(save_handle);
    free_buffers.push(buffer);
  }
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  TIMINGS["load"] += load_time;
  TIMINGS["eval"] += eval_time;
  TIMINGS["save"] += save_time;
}

void set_halide_buf(buffer_t &halide_buf, u8 *buf, size_t size) {
  halide_buf.dev = (uintptr_t) nullptr;

  // "You likely want to set the dev_dirty flag for correctness. (It will
  // not matter if all the code runs on the GPU.)"
  halide_buf.dev_dirty = true;

  i32 err = halide_cuda_wrap_device_ptr(nullptr, &halide_buf, (uintptr_t)buf);
  assert(err == 0);

  // "You'll need to set the host field of the buffer_t structs to
  // something other than nullptr as that is used to indicate bounds query
  // calls" - Zalman Stern
  halide_buf.host = (u8 *)0xdeadbeef;
}

void unset_halide_buf(buffer_t &halide_buf) {
  halide_cuda_detach_device_ptr(nullptr, &halide_buf);
}

void transform_halide(const NetDescriptor& descriptor_,
                      i32 net_width, i32 net_height,
                      u8* input_buffer, u8* output_buffer) {
  i32 net_input_width_ = net_width;
  i32 net_input_height_ = net_height;
  i32 frame_width = width;
  i32 frame_height = height;
  size_t net_input_size =
      net_input_width_ * net_input_height_ * 3 * sizeof(float);

  buffer_t input_buf = {0}, output_buf = {0};

  set_halide_buf(input_buf, input_buffer, frame_width * frame_height * 3);
  set_halide_buf(output_buf, output_buffer, net_input_size);

  // Halide has the input format x * stride[0] + y * stride[1] + c * stride[2]
  // input_buf.host = input_buffer;
  input_buf.stride[0] = 3;
  input_buf.stride[1] = frame_width * 3;
  input_buf.stride[2] = 1;
  input_buf.extent[0] = frame_width;
  input_buf.extent[1] = frame_height;
  input_buf.extent[2] = 3;
  input_buf.elem_size = 1;

  // Halide conveniently defaults to a planar format, which is what Caffe
  // expects
  output_buf.host = output_buffer;
  output_buf.stride[0] = 1;
  output_buf.stride[1] = net_input_width_;
  output_buf.stride[2] = net_input_width_ * net_input_height_;
  output_buf.extent[0] = net_input_width_;
  output_buf.extent[1] = net_input_height_;
  output_buf.extent[2] = 3;
  output_buf.elem_size = 4;

  auto func = caffe_input_transformer_gpu;
  int error = func(&input_buf, frame_width, frame_height,
                   net_input_width_, net_input_height_, descriptor_.normalize,
                   descriptor_.mean_colors[2], descriptor_.mean_colors[1],
                   descriptor_.mean_colors[0], &output_buf);
  LOG_IF(FATAL, error != 0) << "Halide error " << error;

  unset_halide_buf(input_buf);
  unset_halide_buf(output_buf);
}

void video_caffe_worker(int gpu_device_id, Queue<u8 *> &free_buffers,
                        Queue<BufferHandle> &decoded_frames,
                        Queue<SaveHandle> &free_output_buffers,
                        Queue<SaveHandle> &save_buffers) {
  double idle_time = 0;
  double load_time = 0;
  double transform_time = 0;
  double net_time = 0;
  double eval_time = 0;
  double save_time = 0;

  int frame_size = width * height * 4 * sizeof(u8);

  NetDescriptor descriptor;
  {
    //std::ifstream net_file{NET_PATH};
    descriptor = descriptor_from_net_file(NET_PATH);
  }

  // Set ourselves to the correct GPU
  CUcontext cuda_context;
  CUD_CHECK(cuDevicePrimaryCtxRetain(&cuda_context, gpu_device_id));
  Halide::Runtime::Internal::Cuda::context = cuda_context;
  halide_set_gpu_device(gpu_device_id);
  cv::cuda::setDevice(gpu_device_id);
  CU_CHECK(cudaSetDevice(gpu_device_id));
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::Caffe::SetDevice(gpu_device_id);
  std::unique_ptr<caffe::Net<float>> net;
  net.reset(new caffe::Net<float>(descriptor.model_path, caffe::TEST));
  net->CopyTrainedLayersFrom(descriptor.model_weights_path);

  const boost::shared_ptr<caffe::Blob<float>> input_blob{
    net->blob_by_name(descriptor.input_layer_names[0])};

  input_blob->Reshape({NET_BATCH_SIZE, input_blob->shape(1),
                       input_blob->shape(2), input_blob->shape(3)});


  net->Forward();

  int net_input_width = input_blob->shape(2);  // width
  int net_input_height = input_blob->shape(3); // height
  size_t net_input_size =
      net_input_width * net_input_height * 3 * sizeof(float);

  int64_t frame = 0;
  while (true) {
    auto idle_start = scanner::now();
    BufferHandle buffer_handle;
    decoded_frames.pop(buffer_handle);
    u8* buffer = buffer_handle.buffer;
    int elements = buffer_handle.elements;
    if (buffer == nullptr) {
      break;
    }

    SaveHandle save_handle;
    free_output_buffers.pop(save_handle);
    save_handle.elements = elements;

    u8* output_buffer = save_handle.buffer;

    cvc::Stream stream = cvc::StreamAccessor::wrapStream(save_handle.stream);

    idle_time += scanner::nano_since(idle_start);

    // Load the first frame
    int64_t frame = 0;
    while (frame < elements) {
      int batch = std::min((int)(elements - frame), (int)NET_BATCH_SIZE);
      if (batch != NET_BATCH_SIZE) {
        input_blob->Reshape({batch, input_blob->shape(1), input_blob->shape(2),
                input_blob->shape(3)});
      }
      auto transform_start = scanner::now();
      for (int i = 0; i < batch; i++) {
        u8 *input = buffer + (frame + i) * frame_size;
        u8 *output =
            ((u8 *)input_blob->mutable_gpu_data()) + i * net_input_size;
        transform_halide(descriptor, net_input_width, net_input_height,
                         input, output);
      }
      transform_time += scanner::nano_since(transform_start);
      eval_time += scanner::nano_since(transform_start);

      auto net_start = scanner::now();
      net->Forward();
      net_time += scanner::nano_since(net_start);
      eval_time += scanner::nano_since(net_start);

      save_handle.stream = 0;
      // Save outputs
      auto save_start = scanner::now();
      const boost::shared_ptr<caffe::Blob<float>> output_blob{
        net->blob_by_name(descriptor.output_layer_names[0])};
      cudaMemcpyAsync(output_buffer + frame * output_element_size,
                      output_blob->gpu_data(),
                      output_element_size * batch, cudaMemcpyDefault,
                      save_handle.stream);
      frame += batch;
      save_time += scanner::nano_since(save_start);
    }

    save_buffers.push(save_handle);
    free_buffers.push(buffer);
  }
  Halide::Runtime::Internal::Cuda::context = 0;
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  TIMINGS["idle"] += idle_time;
  TIMINGS["load"] += load_time;
  TIMINGS["transform"] += transform_time;
  TIMINGS["net"] += net_time;
  TIMINGS["eval"] += eval_time;
  TIMINGS["save"] += save_time;
}

int main(int argc, char** argv) {
  std::string video_list_path;
  std::string scanner_video_args;
  std::string db_path;
  i32 decoder_count;
  i32 eval_count;
  {
    po::variables_map vm;
    po::options_description desc("Allowed options");
    desc.add_options()("help", "Produce help message")(
        "video_list_path", po::value<std::string>()->required(),
        "Path to video file.")(

        "operation", po::value<std::string>()->required(),
        "histogram, flow, or caffe")(

        "decode_type", po::value<std::string>()->required(),
        "all, stride, gather, range")(

        "decode_args", po::value<std::string>()->required(), "")(

        "db_path", po::value<std::string>()->required(),
        "Path to scanner db for reading preprocessed video files")(

        "scanner_video_args", po::value<std::string>()->required(),
        "Identifiers for scanner video")(

        "decoder_count", po::value<int>()->required(), "Number of decoders")(

        "eval_count", po::value<int>()->required(), "Number of eval routines")(

        "width", po::value<int>()->required(), "Width of video.")(

        "height", po::value<int>()->required(), "Height of video.");

    try {
      po::store(po::parse_command_line(argc, argv, desc), vm);
      po::notify(vm);

      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
      }

      video_list_path = vm["video_list_path"].as<std::string>();
      db_path = vm["db_path"].as<std::string>();
      scanner_video_args = vm["scanner_video_args"].as<std::string>();

      OPERATION = vm["operation"].as<std::string>();
      DECODE_TYPE = vm["decode_type"].as<std::string>();
      DECODE_ARGS = vm["decode_args"].as<std::string>();
      decoder_count = vm["decoder_count"].as<int>();
      eval_count = vm["eval_count"].as<int>();

      width = vm["width"].as<int>();
      height = vm["height"].as<int>();

    } catch (const po::required_option& e) {
      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
      } else {
        throw e;
      }
    }
  }

  scanner::internal::set_database_path(db_path);

  bool cpu_decoder = false;
  bool scanner_decode = true;

  DecoderFn decoder_fn;
  WorkerFn worker_fn;
  if (OPERATION == "decode_cpu" || OPERATION == "stride_cpu" ||
      OPERATION == "gather_cpu" || OPERATION == "range_cpu") {
    BATCH_SIZE = 64;
    cpu_decoder = true;
    scanner_decode = true;
    worker_fn = video_decode_worker;
    output_element_size = 1;
  } else if (OPERATION == "decode_gpu" || OPERATION == "stride_gpu" ||
             OPERATION == "gather_gpu" || OPERATION == "range_gpu") {
    BATCH_SIZE = 64;
    scanner_decode = true;
    worker_fn = video_decode_worker;
    output_element_size = 1;
  } else if (OPERATION == "histogram_cpu") {
    cpu_decoder = true;
    worker_fn = video_histogram_cpu_worker;
    output_element_size = 3 * BINS * sizeof(i32);
  } else if (OPERATION == "histogram_gpu") {
    worker_fn = video_histogram_gpu_worker;
    output_element_size = 3 * BINS * sizeof(i32);
  } else if (OPERATION == "flow_cpu") {
    BATCH_SIZE = 8;      // Batch size for network
    BATCH_SIZE = 1;      // Batch size for network
    cpu_decoder = true;
    scanner_decode = true;
    worker_fn = video_flow_cpu_worker;
    output_element_size = 2 * height * width * sizeof(f32);
  } else if (OPERATION == "flow_gpu") {
    BATCH_SIZE = 8;      // Batch size for network
    worker_fn = video_flow_gpu_worker;
    output_element_size = 2 * height * width * sizeof(f32);
  } else if (OPERATION == "caffe") {
    worker_fn = video_caffe_worker;
    output_element_size = 1000 * sizeof(f32);
  } else {
    exit(1);
  }
  output_buffer_size = BATCH_SIZE * output_element_size;

  IS_CPU = cpu_decoder;
  if (IS_CPU) {
    if (scanner_decode) {
      decoder_fn = scanner_cpu_decoder_worker;
    } else {
      decoder_fn = ffmpeg_decoder_worker;
    }
  } else {
    if (scanner_decode) {
      decoder_fn = scanner_gpu_decoder_worker;
    } else {
      decoder_fn = opencv_decoder_worker;
    }
  }

  // Setup decoder
  CUD_CHECK(cuInit(0));

  volatile bool done = false;

  cudaSetDevice(0);
  // Create decoded frames buffers and output buffers
  i32 queue_size = 1000;
  Queue<u8*> free_buffers(queue_size);
  Queue<BufferHandle> decoded_frames(queue_size);
  Queue<SaveHandle> free_output_buffers(queue_size);
  Queue<SaveHandle> save_buffers(queue_size);
  for (int i = 0; i < NUM_BUFFERS * std::max(decoder_count, eval_count);
       ++i) {
    if (IS_CPU) {
      u8 *buffer;
      buffer = (u8*)malloc(BATCH_SIZE * width * height * 4 * sizeof(u8));
      if (buffer == nullptr) {
        exit(1);
      }
      free_buffers.push(buffer);

      buffer = (u8*)malloc(output_buffer_size);
      if (buffer == nullptr) {
        exit(1);
      }
      SaveHandle handle;
      handle.buffer = buffer;
      free_output_buffers.push(handle);
    } else {
      u8 *buffer;
      CU_CHECK(cudaMalloc((void **)&buffer,
                          BATCH_SIZE * width * height * 4 * sizeof(u8)));
      free_buffers.push(buffer);

      CU_CHECK(cudaMalloc((void **)&buffer, output_buffer_size));
      SaveHandle handle;
      handle.buffer = buffer;
      CU_CHECK(
          cudaStreamCreateWithFlags(&handle.stream, cudaStreamNonBlocking));
      free_output_buffers.push(handle);
    }
  }

  // Start up workers to process videos
  std::vector<std::thread> evaluator_workers;
  for (i32 i = 0; i < eval_count; ++i) {
    evaluator_workers.emplace_back(
        worker_fn, 0, std::ref(free_buffers), std::ref(decoded_frames),
        std::ref(free_output_buffers), std::ref(save_buffers));
  }
  std::thread save_thread(save_worker, 0, std::ref(save_buffers),
                          std::ref(free_output_buffers));

  std::unique_ptr<storehouse::StorageConfig> config(
      storehouse::StorageConfig::make_posix_config());
  std::unique_ptr<storehouse::StorageBackend> storage(
      storehouse::StorageBackend::make_from_config(config.get()));

  // Insert video paths into work queue
  int64_t tid = 0;
  Queue<int64_t> task_ids(10000);
  if (scanner_decode) {
    scanner::proto::MemoryPoolConfig config;
    config.set_pinned_cpu(false);
    config.mutable_cpu()->set_use_pool(false);
    config.mutable_gpu()->set_use_pool(false);
    init_memory_allocators(config, {0});

    // Read index entry

    SCANNER_TABLE_ID = -1;
    SCANNER_COLUMN_ID = -1;
    {
      std::vector<std::string> ids = split(scanner_video_args, ':');
      SCANNER_TABLE_ID = atoi(ids[0].c_str());
      SCANNER_COLUMN_ID = atoi(ids[1].c_str());
    }

    scanner::internal::VideoIndexEntry index_entry = scanner::internal::read_video_index(
        storage.get(), SCANNER_TABLE_ID, SCANNER_COLUMN_ID, 0);

    printf("decode type %s\n", DECODE_TYPE.c_str());
    // Break up work into keyframe sized chunks
    auto& keyframes = index_entry.keyframe_positions;
    if (DECODE_TYPE == "all") {
      for (int i = 1; i < keyframes.size(); ++i) {
        int64_t start = keyframes[i - 1];
        int64_t end = keyframes[i];
        TASK_RANGES.push_back(std::make_tuple(start, end));
      }
    } else if (DECODE_TYPE == "strided") {
      int64_t total_frames = keyframes.back();
      STRIDE = atoi(DECODE_ARGS.c_str());
      int64_t current_frame = STRIDE;
      int keyframe_idx = 1;
      int start_frame = 0;
      while (current_frame < total_frames) {
        if (current_frame >= keyframes[keyframe_idx]) {
          // Insert current keyframe block
          TASK_RANGES.push_back(std::make_tuple(start_frame, current_frame));
          printf("range %ld, %ld\n", start_frame, current_frame);
          start_frame = current_frame;
          // Search for next keyframe start
          int64_t next_frame =
              std::min(current_frame, keyframes.back() - 1);
          for (int i = keyframe_idx; i < keyframes.size(); ++i) {
            printf("keyframe %d\n", keyframes[i]);
            if (next_frame < keyframes[i]) {
              keyframe_idx = i;
              break;
            }
          }
        }
        current_frame += STRIDE;
      }
    } else if (DECODE_TYPE == "range") {
      std::vector<int64_t> range_start;
      std::vector<int64_t> range_end;
      for (auto s : split(DECODE_ARGS, ',')) {
        std::vector<std::string> se = split(s, ':');
        range_start.push_back(atoi(se[0].c_str()));
        range_end.push_back(atoi(se[1].c_str()));
      }

      for (int ri = 0; ri < range_start.size(); ++ri) {
        int64_t rs = range_start[ri];
        int64_t re = range_end[ri];
        int64_t start = 0;
        int64_t end = keyframes[0];
        int i = 0;
        for (; i < keyframes.size(); ++i) {
          if (keyframes[i] > rs) {
            break;
          }
          start = keyframes[i];
        }
        // Start inserting tasks
        for (; i < keyframes.size(); ++i) {
          if (keyframes[i] >= re) {
            break;
          }
          TASK_RANGES.push_back(std::make_tuple(start, keyframes[i]));
          start = keyframes[i];
        }
        TASK_RANGES.push_back(std::make_tuple(start, re));
      }
    }
    for (size_t i = 0; i < TASK_RANGES.size(); ++i) {
      auto& kv = TASK_RANGES[i];
      printf("s/e: %d-%d\n", std::get<0>(kv), std::get<1>(kv));
      task_ids.push(i);
    }

  } else {
    std::ifstream infile(video_list_path);
    std::string line;
    while (std::getline(infile, line)) {
      if (line == "") {
        break;
      }
      std::cout << line << std::endl;
      VIDEO_PATHS.push_back(line);
      task_ids.push(tid++);
    }
  }


  // Wait to make sure everything is setup first
  sleep(5);

  // Start work by setting up feeder
  std::vector<std::thread> decoder_threads;
  for (i32 i = 0; i < decoder_count; ++i) {
    decoder_threads.emplace_back(decoder_fn, 0, std::ref(task_ids),
                                 std::ref(free_buffers),
                                 std::ref(decoded_frames));
  }
  auto total_start = scanner::now();

  for (i32 i = 0; i < decoder_count; ++i) {
    task_ids.push(-1);
  }
  for (i32 i = 0; i < decoder_count; ++i) {
    decoder_threads[i].join();
  }
  // Tell evaluator decoder is done
  for (i32 i = 0; i < eval_count; ++i) {
    BufferHandle empty;
    empty.buffer = nullptr;
    decoded_frames.push(empty);
  }
  for (i32 i = 0; i < eval_count; ++i) {
    evaluator_workers[i].join();
  }

  SaveHandle em;
  em.buffer = nullptr;
  save_buffers.push(em);
  save_thread.join();

  sync();
  TIMINGS["total"] = scanner::nano_since(total_start);

  for (auto& kv : TIMINGS) {
    printf("TIMING: %s,%.2f\n", kv.first.c_str(), kv.second / 1000000000.0);
  }
}
