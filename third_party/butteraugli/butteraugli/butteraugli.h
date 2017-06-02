// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Disclaimer: This is not an official Google product.
//
// Author: Jyrki Alakuijala (jyrki.alakuijala@gmail.com)

#ifndef BUTTERAUGLI_BUTTERAUGLI_H_
#define BUTTERAUGLI_BUTTERAUGLI_H_

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#ifndef PROFILER_ENABLED
#define PROFILER_ENABLED 0
#endif
#if PROFILER_ENABLED
#else
#define PROFILER_FUNC
#define PROFILER_ZONE(name)
#endif

namespace butteraugli {

class ButteraugliComparator {
 public:
  ButteraugliComparator(size_t xsize, size_t ysize, int step);

  // Computes the butteraugli map between xyb0 and xyb1 and updates result.
  // Both xyb0 and xyb1 are in opsin-dynamics space.
  // NOTE: The xyb0 and xyb1 images are mutated by this function in-place.
  void DiffmapOpsinDynamicsImage(std::vector<std::vector<float>> &xyb0,
                                 std::vector<std::vector<float>> &xyb1,
                                 std::vector<float> &result);

 private:
  void BlockDiffMap(const std::vector<std::vector<float> > &rgb0,
                    const std::vector<std::vector<float> > &rgb1,
                    std::vector<float>* block_diff_dc,
                    std::vector<float>* block_diff_ac);


  void EdgeDetectorMap(const std::vector<std::vector<float> > &rgb0,
                       const std::vector<std::vector<float> > &rgb1,
                       std::vector<float>* edge_detector_map);

  void EdgeDetectorLowFreq(const std::vector<std::vector<float> > &rgb0,
                           const std::vector<std::vector<float> > &rgb1,
                           std::vector<float>* block_diff_ac);

  void CombineChannels(const std::vector<std::vector<float> >& scale_xyb,
                       const std::vector<std::vector<float> >& scale_xyb_dc,
                       const std::vector<float>& block_diff_dc,
                       const std::vector<float>& block_diff_ac,
                       const std::vector<float>& edge_detector_map,
                       std::vector<float>* result);

  const size_t xsize_;
  const size_t ysize_;
  const size_t num_pixels_;
  const int step_;
  const size_t res_xsize_;
  const size_t res_ysize_;
};

double ButteraugliScoreFromDiffmap(const std::vector<float>& distmap);

// Compute values of local frequency and dc masking based on the activity
// in the two images.
void Mask(const std::vector<std::vector<float> > &rgb0,
          const std::vector<std::vector<float> > &rgb1,
          size_t xsize, size_t ysize,
          std::vector<std::vector<float> > *mask,
          std::vector<std::vector<float> > *mask_dc);

// Computes difference metrics for one 8x8 block.
void ButteraugliBlockDiff(double rgb0[192],
                          double rgb1[192],
                          double diff_xyb_dc[3],
                          double diff_xyb_ac[3],
                          double diff_xyb_edge_dc[3]);

void OpsinDynamicsImage(size_t xsize, size_t ysize,
                        std::vector<std::vector<float> > &rgb);

void MaskHighIntensityChange(
    size_t xsize, size_t ysize,
    const std::vector<std::vector<float> > &c0,
    const std::vector<std::vector<float> > &c1,
    std::vector<std::vector<float> > &rgb0,
    std::vector<std::vector<float> > &rgb1);

void Blur(size_t xsize, size_t ysize, float* channel, double sigma,
          double border_ratio = 0.0);

}  // namespace butteraugli

#endif  // BUTTERAUGLI_BUTTERAUGLI_H_
