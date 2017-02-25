#include "net_descriptor.h"

#include "caffe/blob.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "toml/toml.h"
#include <vector>

NetDescriptor
descriptor_from_net_file(const std::string &net_file_path) {
  std::ifstream net_file{net_file_path};

  toml::ParseResult pr = toml::parse(net_file);
  if (!pr.valid()) {
    LOG(FATAL) << pr.errorReason;
  }
  const toml::Value &root = pr.value;

  NetDescriptor descriptor;

  auto net = root.find("net");
  if (!net) {
    std::cout << "Missing 'net': net description map" << std::endl;
    exit(EXIT_FAILURE);
  }

  auto model_path = net->find("model");
  if (!model_path) {
    std::cout << "Missing 'net.model': path to model" << std::endl;
    exit(EXIT_FAILURE);
  }
  auto weights_path = net->find("weights");
  if (!weights_path) {
    std::cout << "Missing 'net.weights': path to model weights" << std::endl;
    exit(EXIT_FAILURE);
  }
  auto input_layers = net->find("input_layers");
  if (!input_layers) {
    std::cout << "Missing 'net.input_layers': name of input layers "
              << std::endl;
    exit(EXIT_FAILURE);
  }
  auto output_layers = net->find("output_layers");
  if (!output_layers) {
    std::cout << "Missing 'net.output_layers': name of output layers "
              << std::endl;
    exit(EXIT_FAILURE);
  }
  auto input_format = net->find("input");
  if (!input_format) {
    std::cout << "Missing 'net.input': description of net input format "
              << std::endl;
    exit(EXIT_FAILURE);
  }
  auto dimensions_ordering = input_format->find("dimensions");
  if (!dimensions_ordering) {
    std::cout << "Missing 'net.input.dimensions': ordering of dimensions "
              << "for input format " << std::endl;
    exit(EXIT_FAILURE);
  }
  auto channel_ordering = input_format->find("channel_ordering");
  if (!channel_ordering) {
    std::cout << "Missing 'net.input.channel_ordering': ordering of channels "
              << "for input format " << std::endl;
    exit(EXIT_FAILURE);
  }

  descriptor.model_path = model_path->as<std::string>();
  descriptor.model_weights_path = weights_path->as<std::string>();
  for (const toml::Value &v : input_layers->as<toml::Array>()) {
    descriptor.input_layer_names.push_back(v.as<std::string>());
  }
  for (const toml::Value &v : output_layers->as<toml::Array>()) {
    descriptor.output_layer_names.push_back(v.as<std::string>());
  }

  auto input_width = net->find("input_width");
  auto input_height = net->find("input_height");
  auto preserve_aspect_ratio = net->find("preserve_aspect_ratio");
  bool preserve_aspect = false;
  if (preserve_aspect_ratio) {
    preserve_aspect = preserve_aspect_ratio->as<bool>();
  }
  //descriptor.set_preserve_aspect_ratio(preserve_aspect);

  //descriptor.set_input_width(-1);
  //descriptor.set_input_height(-1);
  if (preserve_aspect) {
    if (input_height) {
      //descriptor.set_input_height(input_height->as<i32>());
    } else if (input_width) {
      //descriptor.set_input_width(input_width->as<i32>());
    } else {
      std::cout << "'preserve_aspect_ratio': must specify only one of "
                   "input_width or input_height"
                << std::endl;
      exit(EXIT_FAILURE);
    }
  } else if (input_width && input_height) {
    //descriptor.set_input_width(input_width->as<i32>());
    //descriptor.set_input_height(input_height->as<i32>());
  }

  auto pad_mod = net->find("pad_mod");
  //descriptor.set_pad_mod(pad_mod ? pad_mod->as<i32>() : -1);

  auto normalize = net->find("normalize");
  descriptor.normalize = normalize ? normalize->as<bool>() : false;

  auto transpose = net->find("transpose");
  //descriptor.set_transpose(transpose ? transpose->as<bool>() : false);

  auto mean_image = root.find("mean-image");
  if (!mean_image) {
    std::cout << "Missing 'mean-image': mean image descripton map" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (mean_image->has("colors")) {
    auto mean_blue = mean_image->find("colors.blue");
    if (!mean_blue) {
      std::cout << "Missing 'mean-image.colors.blue'" << std::endl;
      exit(EXIT_FAILURE);
    }
    auto mean_green = mean_image->find("colors.green");
    if (!mean_green) {
      std::cout << "Missing 'mean-image.colors.green'" << std::endl;
      exit(EXIT_FAILURE);
    }
    auto mean_red = mean_image->find("colors.red");
    if (!mean_red) {
      std::cout << "Missing 'mean-image.colors.red'" << std::endl;
      exit(EXIT_FAILURE);
    }

    float blue = mean_blue->as<double>();
    float green = mean_green->as<double>();
    float red = mean_red->as<double>();

    for (const toml::Value &v : channel_ordering->as<toml::Array>()) {
      std::string color = v.as<std::string>();
      if (color == "red") {
        descriptor.mean_colors.push_back(red);
      } else if (color == "green") {
        descriptor.mean_colors.push_back(green);
      } else if (color == "blue") {
        descriptor.mean_colors.push_back(blue);
      }
    }
  } else if (mean_image->has("path")) {
    std::string mean_path = mean_image->get<std::string>("path");

    auto mean_image_width = mean_image->find("width");
    if (!mean_image_width) {
      std::cout << "Missing 'mean-image.width': width of mean" << std::endl;
      exit(EXIT_FAILURE);
    }
    auto mean_image_height = mean_image->find("height");
    if (!mean_image_height) {
      std::cout << "Missing 'mean-image.height': height of mean" << std::endl;
      exit(EXIT_FAILURE);
    }

    /*
    descriptor.set_mean_width(mean_image_width->as<int>());
    descriptor.set_mean_height(mean_image_height->as<int>());

    int mean_size = descriptor.mean_width() * descriptor.mean_height();

    // Load mean image
    Blob<float> data_mean;
    BlobProto blob_proto;
    bool result = ReadProtoFromBinaryFile(mean_path, &blob_proto);
    data_mean.FromProto(blob_proto);

    memcpy(descriptor.mutable_mean_image(), data_mean.cpu_data(),
           sizeof(float) * mean_size * 3);
    */
  } else if (!mean_image->has("empty")) {
    std::cout << "Missing 'mean-image.{colors,path,empty}': must specify "
              << "color channel values or path of mean image file or that "
              << "there is no mean" << std::endl;
    exit(EXIT_FAILURE);
  }

  return descriptor;
}
