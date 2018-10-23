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

namespace {
using ::butteraugli::ImageF;
using ::butteraugli::CreatePlanes;
using ::butteraugli::PlanesFromPacked;
using ::butteraugli::PackedFromPlanes;

std::vector<ImageF> LinearRgb(const size_t xsize, const size_t ysize,
                              const std::vector<uint8_t>& rgb) {
  const double* lut = Srgb8ToLinearTable();
  std::vector<ImageF> planes = CreatePlanes<float>(xsize, ysize, 3);
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      const uint8_t* const BUTTERAUGLI_RESTRICT row_in = &rgb[3 * xsize * y];
      float* const BUTTERAUGLI_RESTRICT row_out = planes[c].Row(y);
      for (size_t x = 0; x < xsize; ++x) {
        row_out[x] = lut[row_in[3 * x + c]];
      }
    }
  }
  return planes;
}

}  // namespace

ButteraugliComparator::ButteraugliComparator(const int width, const int height,
                                             const std::vector<uint8_t>* rgb,
                                             const float target_distance,
                                             ProcessStats* stats)
    : width_(width),
      height_(height),
      target_distance_(target_distance),
      rgb_orig_(*rgb),
      comparator_(LinearRgb(width_, height_, *rgb)),
      distance_(0.0),
      stats_(stats) {}

void ButteraugliComparator::Compare(const OutputImage& img) {
  std::vector<ImageF> rgb0 =
      ::butteraugli::OpsinDynamicsImage(LinearRgb(width_, height_, rgb_orig_));
  std::vector<std::vector<float> > rgb(3, std::vector<float>(width_ * height_));
  img.ToLinearRGB(&rgb);
  const std::vector<ImageF> rgb_planes = PlanesFromPacked(width_, height_, rgb);
  std::vector<float>(width_ * height_).swap(distmap_);
  ImageF distmap;
  comparator_.Diffmap(rgb_planes, distmap);
  CopyToPacked(distmap, &distmap_);
  distance_ = ::butteraugli::ButteraugliScoreFromDiffmap(distmap);
  GUETZLI_LOG(stats_, " BA[100.00%%] D[%6.4f]", distance_);
}

namespace {

// To change this to n, add the relevant FFTn function and kFFTnMapIndexTable.
constexpr size_t kBlockEdge = 8;
constexpr size_t kBlockSize = kBlockEdge * kBlockEdge;
constexpr size_t kBlockEdgeHalf = kBlockEdge / 2;
constexpr size_t kBlockHalf = kBlockEdge * kBlockEdgeHalf;

// Contrast sensitivity related weights.
// Contrast sensitivity matrix is a model of the contrast sensitivity of the
// human eye. The contrast sensitivity function, also known as contrast
// sensitivity curve, is a basic psychovisual function and it roughly means
// that highest frequences have less impact than middle and low frequences.
//
// The order of coefficients here is slightly confusing.
// It is a creative order that benefits from the mirroring of the fft.
static const double *GetContrastSensitivityMatrix() {
  static double csf8x8[kBlockHalf + kBlockEdgeHalf + 1] = {
    0.0,  // no 8x8 dc, enough 'dc' is calculated from low/mid freq images.
    0.0,
    0.0,
    0.0,
    0.3831134973,
    0.676303603859,
    1.1550451483,
    8,
    8,
    0.692062533689,
    0.847511538605,
    0.498250875965,
    0.36198671102,
    0.308982169883,
    0.1312701920435,
    4.71274312228,
    1.1550451483,
    0.847511538605,
    4.71274312228,
    0.991205724152,
    1.30229591239,
    0.627264168628,
    0.4,
    0.1312701920435,
    0.676303603859,
    0.498250875965,
    0.991205724152,
    0.5,
    0.3831134973,
    0.349686450518,
    0.627264168628,
    0.308982169883,
    0.3831134973,
    0.36198671102,
    1.30229591239,
    0.3831134973,
    0.323078800177,
  };
  return &csf8x8[0];
}

struct Complex {
  double real;
  double imag;
};

inline double abssq(const Complex& c) {
  return c.real * c.real + c.imag * c.imag;
}

static void TransposeBlock(Complex data[kBlockSize]) {
  for (int i = 0; i < kBlockEdge; i++) {
    for (int j = 0; j < i; j++) {
      std::swap(data[kBlockEdge * i + j], data[kBlockEdge * j + i]);
    }
  }
}

//  D. J. Bernstein's Fast Fourier Transform algorithm on 4 elements.
inline void FFT4(Complex* a) {
  double t1, t2, t3, t4, t5, t6, t7, t8;
  t5 = a[2].real;
  t1 = a[0].real - t5;
  t7 = a[3].real;
  t5 += a[0].real;
  t3 = a[1].real - t7;
  t7 += a[1].real;
  t8 = t5 + t7;
  a[0].real = t8;
  t5 -= t7;
  a[1].real = t5;
  t6 = a[2].imag;
  t2 = a[0].imag - t6;
  t6 += a[0].imag;
  t5 = a[3].imag;
  a[2].imag = t2 + t3;
  t2 -= t3;
  a[3].imag = t2;
  t4 = a[1].imag - t5;
  a[3].real = t1 + t4;
  t1 -= t4;
  a[2].real = t1;
  t5 += a[1].imag;
  a[0].imag = t6 + t5;
  t6 -= t5;
  a[1].imag = t6;
}

const double kSqrtHalf = 0.70710678118654752440084436210484903;

//  D. J. Bernstein's Fast Fourier Transform algorithm on 8 elements.
void FFT8(Complex* a) {
  double t1, t2, t3, t4, t5, t6, t7, t8;

  t7 = a[4].imag;
  t4 = a[0].imag - t7;
  t7 += a[0].imag;
  a[0].imag = t7;

  t8 = a[6].real;
  t5 = a[2].real - t8;
  t8 += a[2].real;
  a[2].real = t8;

  t7 = a[6].imag;
  a[6].imag = t4 - t5;
  t4 += t5;
  a[4].imag = t4;

  t6 = a[2].imag - t7;
  t7 += a[2].imag;
  a[2].imag = t7;

  t8 = a[4].real;
  t3 = a[0].real - t8;
  t8 += a[0].real;
  a[0].real = t8;

  a[4].real = t3 - t6;
  t3 += t6;
  a[6].real = t3;

  t7 = a[5].real;
  t3 = a[1].real - t7;
  t7 += a[1].real;
  a[1].real = t7;

  t8 = a[7].imag;
  t6 = a[3].imag - t8;
  t8 += a[3].imag;
  a[3].imag = t8;
  t1 = t3 - t6;
  t3 += t6;

  t7 = a[5].imag;
  t4 = a[1].imag - t7;
  t7 += a[1].imag;
  a[1].imag = t7;

  t8 = a[7].real;
  t5 = a[3].real - t8;
  t8 += a[3].real;
  a[3].real = t8;

  t2 = t4 - t5;
  t4 += t5;

  t6 = t1 - t4;
  t8 = kSqrtHalf;
  t6 *= t8;
  a[5].real = a[4].real - t6;
  t1 += t4;
  t1 *= t8;
  a[5].imag = a[4].imag - t1;
  t6 += a[4].real;
  a[4].real = t6;
  t1 += a[4].imag;
  a[4].imag = t1;

  t5 = t2 - t3;
  t5 *= t8;
  a[7].imag = a[6].imag - t5;
  t2 += t3;
  t2 *= t8;
  a[7].real = a[6].real - t2;
  t2 += a[6].real;
  a[6].real = t2;
  t5 += a[6].imag;
  a[6].imag = t5;

  FFT4(a);

  // Reorder to the correct output order.
  // TODO(szabadka): Modify the above computation so that this is not needed.
  Complex tmp = a[2];
  a[2] = a[3];
  a[3] = a[5];
  a[5] = a[7];
  a[7] = a[4];
  a[4] = a[1];
  a[1] = a[6];
  a[6] = tmp;
}

// Same as FFT8, but all inputs are real.
// TODO(szabadka): Since this does not need to be in-place, maybe there is a
// faster FFT than this one, which is derived from DJB's in-place complex FFT.
void RealFFT8(const double* in, Complex* out) {
  double t1, t2, t3, t5, t6, t7, t8;
  t8 = in[6];
  t5 = in[2] - t8;
  t8 += in[2];
  out[2].real = t8;
  out[6].imag = -t5;
  out[4].imag = t5;
  t8 = in[4];
  t3 = in[0] - t8;
  t8 += in[0];
  out[0].real = t8;
  out[4].real = t3;
  out[6].real = t3;
  t7 = in[5];
  t3 = in[1] - t7;
  t7 += in[1];
  out[1].real = t7;
  t8 = in[7];
  t5 = in[3] - t8;
  t8 += in[3];
  out[3].real = t8;
  t2 = -t5;
  t6 = t3 - t5;
  t8 = kSqrtHalf;
  t6 *= t8;
  out[5].real = out[4].real - t6;
  t1 = t3 + t5;
  t1 *= t8;
  out[5].imag = out[4].imag - t1;
  t6 += out[4].real;
  out[4].real = t6;
  t1 += out[4].imag;
  out[4].imag = t1;
  t5 = t2 - t3;
  t5 *= t8;
  out[7].imag = out[6].imag - t5;
  t2 += t3;
  t2 *= t8;
  out[7].real = out[6].real - t2;
  t2 += out[6].real;
  out[6].real = t2;
  t5 += out[6].imag;
  out[6].imag = t5;
  t5 = out[2].real;
  t1 = out[0].real - t5;
  t7 = out[3].real;
  t5 += out[0].real;
  t3 = out[1].real - t7;
  t7 += out[1].real;
  t8 = t5 + t7;
  out[0].real = t8;
  t5 -= t7;
  out[1].real = t5;
  out[2].imag = t3;
  out[3].imag = -t3;
  out[3].real = t1;
  out[2].real = t1;
  out[0].imag = 0;
  out[1].imag = 0;

  // Reorder to the correct output order.
  // TODO(szabadka): Modify the above computation so that this is not needed.
  Complex tmp = out[2];
  out[2] = out[3];
  out[3] = out[5];
  out[5] = out[7];
  out[7] = out[4];
  out[4] = out[1];
  out[1] = out[6];
  out[6] = tmp;
}

// Fills in block[kBlockEdgeHalf..(kBlockHalf+kBlockEdgeHalf)], and leaves the
// rest unmodified.
void ButteraugliFFTSquared(double block[kBlockSize]) {
  double global_mul = 0.000064;
  Complex block_c[kBlockSize];
  assert(kBlockEdge == 8);
  for (int y = 0; y < kBlockEdge; ++y) {
    RealFFT8(block + y * kBlockEdge, block_c + y * kBlockEdge);
  }
  TransposeBlock(block_c);
  double r0[kBlockEdge];
  double r1[kBlockEdge];
  for (int x = 0; x < kBlockEdge; ++x) {
    r0[x] = block_c[x].real;
    r1[x] = block_c[kBlockHalf + x].real;
  }
  RealFFT8(r0, block_c);
  RealFFT8(r1, block_c + kBlockHalf);
  for (int y = 1; y < kBlockEdgeHalf; ++y) {
    FFT8(block_c + y * kBlockEdge);
  }
  for (int i = kBlockEdgeHalf; i < kBlockHalf + kBlockEdgeHalf + 1; ++i) {
    block[i] = abssq(block_c[i]);
    block[i] *= global_mul;
  }
}

void ButteraugliBlockDiff(double xyb0[3 * kBlockSize],
                          double xyb1[3 * kBlockSize],
                          double* diff_xyb) {
  const double *csf8x8 = GetContrastSensitivityMatrix();
  double avgdiff_xyb[3] = {0.0};
  for (int i = 0; i < 3 * kBlockSize; ++i) {
    avgdiff_xyb[i / kBlockSize] += xyb0[i] - xyb1[i];
  }
  for (int c = 0; c < 3; ++c) {
    double avgdiff = avgdiff_xyb[c] / kBlockSize;
    diff_xyb[c] += 4.0 * avgdiff * avgdiff;
  }
  double x_diff[kBlockSize];
  double y_diff[kBlockSize];
  double b_diff[kBlockSize];
  for (int i = 0; i < kBlockSize; ++i) {
    x_diff[i] = (xyb0[i] - xyb1[i]);
    y_diff[i] = (xyb0[kBlockSize + i] - xyb1[kBlockSize + i]);
    b_diff[i] = (xyb0[2 * kBlockSize + i] - xyb1[2 * kBlockSize + i]);
  }
  ButteraugliFFTSquared(x_diff);
  ButteraugliFFTSquared(y_diff);
  ButteraugliFFTSquared(b_diff);
  for (size_t i = kBlockEdgeHalf; i < kBlockHalf + kBlockEdgeHalf + 1; ++i) {
    double d = csf8x8[i];
    diff_xyb[0] += d * x_diff[i];
    diff_xyb[1] += d * y_diff[i];
    diff_xyb[2] += d * b_diff[i];
  }
}

}  // namespace

void ButteraugliComparator::StartBlockComparisons() {
  std::vector<ImageF> dummy(3);
  std::vector<ImageF> rgb0 =
      ::butteraugli::OpsinDynamicsImage(LinearRgb(width_, height_, rgb_orig_));
  ::butteraugli::Mask(rgb0, rgb0,
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
      per_block_pregamma_[bx] =
          ::butteraugli::PackedFromPlanes(::butteraugli::OpsinDynamicsImage(
              ::butteraugli::PlanesFromPacked(8, 8, per_block_pregamma_[bx])));
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
  const std::vector<std::vector<float> >& rgb0 =
      per_block_pregamma_[block_ix];

  std::vector<std::vector<float> > rgb1(3, std::vector<float>(kDCTBlockSize));
  img.ToLinearRGB(xmin, ymin, 8, 8, &rgb1);
  rgb1 = ::butteraugli::PackedFromPlanes(::butteraugli::OpsinDynamicsImage(
      ::butteraugli::PlanesFromPacked(8, 8, rgb1)));

  double b0[3 * kDCTBlockSize];
  double b1[3 * kDCTBlockSize];
  for (int c = 0; c < 3; ++c) {
    for (int ix = 0; ix < kDCTBlockSize; ++ix) {
      b0[c * kDCTBlockSize + ix] = rgb0[c][ix];
      b1[c * kDCTBlockSize + ix] = rgb1[c][ix];
    }
  }
  double diff_xyz[3] = { 0.0 };
  ButteraugliBlockDiff(b0, b1, diff_xyz);

  double diff = 0.0;
  for (int c = 0; c < 3; ++c) {
    diff += diff_xyz[c] * mask_xyz_[c].Row(ymin)[xmin];
  }
  return sqrt(diff);
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
