#pragma once

#include <chrono>

namespace scanner {

using timepoint_t = std::chrono::time_point<std::chrono::high_resolution_clock>;

inline timepoint_t now() { return std::chrono::high_resolution_clock::now(); }

inline double nano_since(timepoint_t then) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now() - then)
      .count();
}

}
