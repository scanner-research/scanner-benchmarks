#pragma once

#include <string>
#include <vector>

struct NetDescriptor {
  std::string model_path;
  std::string model_weights_path;
  std::vector<std::string> input_layer_names;
  std::vector<std::string> output_layer_names;
  std::vector<float> mean_colors;
  bool normalize;
};

NetDescriptor descriptor_from_net_file(const std::string& path);
