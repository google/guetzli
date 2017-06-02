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

#include "guetzli/butteraugli_comparator.h"

#include <algorithm>

#include "guetzli/debug_print.h"
#include "guetzli/gamma_correct.h"
#include "guetzli/score.h"

namespace guetzli {

std::vector<std::vector<float> > ComputeOpsinDynamicsImage(
    const int width, const int height,
    const std::vector<uint8_t>& rgb) {
  std::vector<std::vector<float> > out(3);
  const double* lut = Srgb8ToLinearTable();
  for (int c = 0; c < 3; ++c) {
    out[c].resize(width * height);
    for (int y = 0, ix = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x, ++ix) {
        out[c][ix] = lut[rgb[3 * ix + c]];
      }
    }
  }
  ::butteraugli::OpsinDynamicsImage(width, height, out);
  return out;
}

ButteraugliComparator::ButteraugliComparator(const int width, const int height,
                                             const std::vector<uint8_t>* rgb,
                                             const float target_distance,
                                             ProcessStats* stats)
    : width_(width),
      height_(height),
      target_distance_(target_distance),
      rgb_orig_(*rgb),
      comparator_(width_, height_, kButteraugliStep),
      distance_(0.0),
      stats_(stats) {}

void ButteraugliComparator::Compare(const OutputImage& img) {
  std::vector<std::vector<float> > rgb0 =
      ComputeOpsinDynamicsImage(width_, height_, rgb_orig_);
  std::vector<std::vector<float> > rgb(3, std::vector<float>(width_ * height_));
  img.ToLinearRGB(&rgb);
  ::butteraugli::OpsinDynamicsImage(width_, height_, rgb);
  std::vector<float>().swap(distmap_);
  comparator_.DiffmapOpsinDynamicsImage(rgb0, rgb, distmap_);
  distance_ = ::butteraugli::ButteraugliScoreFromDiffmap(distmap_);
  GUETZLI_LOG(stats_, " BA[100.00%%] D[%6.4f]", distance_);
}

void ButteraugliComparator::StartBlockComparisons() {
  std::vector<std::vector<float> > dummy(3);
  std::vector<std::vector<float> > rgb0 =
      ComputeOpsinDynamicsImage(width_, height_, rgb_orig_);
  ::butteraugli::Mask(rgb0, rgb0,
                      width_, height_,
                      &mask_xyz_, &dummy);
}

void ButteraugliComparator::FinishBlockComparisons() {
  mask_xyz_.clear();
}

void ButteraugliComparator::SwitchBlock(int block_x, int block_y,
                                        int factor_x, int factor_y) {
  block_x_ = block_x;
  block_y_ = block_y;
  factor_x_ = factor_x;
  factor_y_ = factor_y;
  per_block_pregamma_.resize(factor_x_ * factor_y_);
  const double* lut = Srgb8ToLinearTable();
  for (int off_y = 0, bx = 0; off_y < factor_y_; ++off_y) {
    for (int off_x = 0; off_x < factor_x_; ++off_x, ++bx) {
      per_block_pregamma_[bx].resize(3, std::vector<float>(kDCTBlockSize));
      int block_xx = block_x_ * factor_x_ + off_x;
      int block_yy = block_y_ * factor_y_ + off_y;
      for (int iy = 0, i = 0; iy < 8; ++iy) {
        for (int ix = 0; ix < 8; ++ix, ++i) {
          int x = std::min(8 * block_xx + ix, width_ - 1);
          int y = std::min(8 * block_yy + iy, height_ - 1);
          int px = y * width_ + x;
          for (int c = 0; c < 3; ++c) {
            per_block_pregamma_[bx][c][i] = lut[rgb_orig_[3 * px + c]];
          }
        }
      }
      ::butteraugli::OpsinDynamicsImage(8, 8, per_block_pregamma_[bx]);
    }
  }
}

double ButteraugliComparator::CompareBlock(const OutputImage& img,
                                           int off_x, int off_y) const {
  int block_x = block_x_ * factor_x_ + off_x;
  int block_y = block_y_ * factor_y_ + off_y;
  int xmin = 8 * block_x;
  int ymin = 8 * block_y;
  int block_ix = off_y * factor_x_ + off_x;
  const std::vector<std::vector<float> >& rgb0_c =
      per_block_pregamma_[block_ix];

  std::vector<std::vector<float> > rgb1_c(3, std::vector<float>(kDCTBlockSize));
  img.ToLinearRGB(xmin, ymin, 8, 8, &rgb1_c);
  ::butteraugli::OpsinDynamicsImage(8, 8, rgb1_c);

  std::vector<std::vector<float> > rgb0 = rgb0_c;
  std::vector<std::vector<float> > rgb1 = rgb1_c;

  ::butteraugli::MaskHighIntensityChange(8, 8, rgb0_c, rgb1_c, rgb0, rgb1);

  double b0[3 * kDCTBlockSize];
  double b1[3 * kDCTBlockSize];
  for (int c = 0; c < 3; ++c) {
    for (int ix = 0; ix < kDCTBlockSize; ++ix) {
      b0[c * kDCTBlockSize + ix] = rgb0[c][ix];
      b1[c * kDCTBlockSize + ix] = rgb1[c][ix];
    }
  }
  double diff_xyz_dc[3] = { 0.0 };
  double diff_xyz_ac[3] = { 0.0 };
  double diff_xyz_edge_dc[3] = { 0.0 };
  ::butteraugli::ButteraugliBlockDiff(
       b0, b1, diff_xyz_dc, diff_xyz_ac, diff_xyz_edge_dc);

  double scale[3];
  for (int c = 0; c < 3; ++c) {
    scale[c] = mask_xyz_[c][ymin * width_ + xmin];
  }

  static const double kEdgeWeight = 0.05;

  double diff = 0.0;
  double diff_edge = 0.0;
  for (int c = 0; c < 3; ++c) {
    diff += diff_xyz_dc[c] * scale[c];
    diff += diff_xyz_ac[c] * scale[c];
    diff_edge += diff_xyz_edge_dc[c] * scale[c];
  }
  return sqrt((1 - kEdgeWeight) * diff + kEdgeWeight * diff_edge);
}

float ButteraugliComparator::BlockErrorLimit() const {
  return target_distance_;
}

void ButteraugliComparator::ComputeBlockErrorAdjustmentWeights(
      int direction,
      int max_block_dist,
      double target_mul,
      int factor_x, int factor_y,
      const std::vector<float>& distmap,
      std::vector<float>* block_weight) {
  const double target_distance = target_distance_ * target_mul;
  const int sizex = 8 * factor_x;
  const int sizey = 8 * factor_y;
  const int block_width = (width_ + sizex - 1) / sizex;
  const int block_height = (height_ + sizey - 1) / sizey;
  std::vector<float> max_dist_per_block(block_width * block_height);
  for (int block_y = 0; block_y < block_height; ++block_y) {
    for (int block_x = 0; block_x < block_width; ++block_x) {
      int block_ix = block_y * block_width + block_x;
      int x_max = std::min(width_, sizex * (block_x + 1));
      int y_max = std::min(height_, sizey * (block_y + 1));
      float max_dist = 0.0;
      for (int y = sizey * block_y; y < y_max; ++y) {
        for (int x = sizex * block_x; x < x_max; ++x) {
          max_dist = std::max(max_dist, distmap[y * width_ + x]);
        }
      }
      max_dist_per_block[block_ix] = max_dist;
    }
  }
  for (int block_y = 0; block_y < block_height; ++block_y) {
    for (int block_x = 0; block_x < block_width; ++block_x) {
      int block_ix = block_y * block_width + block_x;
      float max_local_dist = static_cast<float>(target_distance);
      int x_min = std::max(0, block_x - max_block_dist);
      int y_min = std::max(0, block_y - max_block_dist);
      int x_max = std::min(block_width, block_x + 1 + max_block_dist);
      int y_max = std::min(block_height, block_y + 1 + max_block_dist);
      for (int y = y_min; y < y_max; ++y) {
        for (int x = x_min; x < x_max; ++x) {
          max_local_dist =
              std::max(max_local_dist, max_dist_per_block[y * block_width + x]);
        }
      }
      if (direction > 0) {
        if (max_dist_per_block[block_ix] <= target_distance &&
            max_local_dist <= 1.1 * target_distance) {
          (*block_weight)[block_ix] = 1.0;
        }
      } else {
        constexpr double kLocalMaxWeight = 0.5;
        if (max_dist_per_block[block_ix] <=
            (1 - kLocalMaxWeight) * target_distance +
            kLocalMaxWeight * max_local_dist) {
          continue;
        }
        for (int y = y_min; y < y_max; ++y) {
          for (int x = x_min; x < x_max; ++x) {
            int d = std::max(std::abs(y - block_y), std::abs(x - block_x));
            int ix = y * block_width + x;
            (*block_weight)[ix] = std::max<float>(
                (*block_weight)[ix], 1.0f / (d + 1.0f));
          }
        }
      }
    }
  }
}

double ButteraugliComparator::ScoreOutputSize(int size) const {
  return ScoreJPEG(distance_, size, target_distance_);
}


}  // namespace guetzli
