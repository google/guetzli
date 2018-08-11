/*
 * Copyright 2016 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GUETZLI_BUTTERAUGLI_COMPARATOR_H_
#define GUETZLI_BUTTERAUGLI_COMPARATOR_H_

#include <vector>

#include "butteraugli/butteraugli.h"
#include "guetzli/comparator.h"
#include "guetzli/jpeg_data.h"
#include "guetzli/output_image.h"
#include "guetzli/stats.h"

namespace guetzli {

constexpr int kButteraugliStep = 3;

class ButteraugliComparator : public Comparator {
 public:
  ButteraugliComparator(const int width, const int height,
                        const std::vector<uint8_t>* rgb,
                        const float target_distance, ProcessStats* stats);

  void Compare(const OutputImage& img) override;

  void StartBlockComparisons() override;
  void FinishBlockComparisons() override;

  void SwitchBlock(int block_x, int block_y,
                   int factor_x, int factor_y) override;

  double CompareBlock(const OutputImage& img,
                      int off_x, int off_y) const override;

  double ScoreOutputSize(int size) const override;

  bool DistanceOK(double target_mul) const override {
    return distance_ <= target_mul * target_distance_;
  }

  const std::vector<float> distmap() const override { return distmap_; }
  float distmap_aggregate() const override { return distance_; }

  float BlockErrorLimit() const override;

  void ComputeBlockErrorAdjustmentWeights(
      int direction, int max_block_dist, double target_mul, int factor_x,
      int factor_y, const std::vector<float>& distmap,
      std::vector<float>* block_weight) override;

 private:
  const int width_;
  const int height_;
  const float target_distance_;
  const std::vector<uint8_t>& rgb_orig_;
  int block_x_;
  int block_y_;
  int factor_x_;
  int factor_y_;
  std::vector<::butteraugli::ImageF> mask_xyz_;
  std::vector<std::vector<std::vector<float>>> per_block_pregamma_;
  ::butteraugli::ButteraugliComparator comparator_;
  float distance_;
  std::vector<float> distmap_;
  ProcessStats* stats_;
};

}  // namespace guetzli

#endif  // GUETZLI_BUTTERAUGLI_COMPARATOR_H_
