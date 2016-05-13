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
// Author: Jyrki Alakuijala (jyrki.alakuijala@gmail.com)

#include "butteraugli.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>

namespace butteraugli {

static const double kInternalGoodQualityThreshold = 12.84;
static const double kGlobalScale = 1.0 / kInternalGoodQualityThreshold;

inline double DotProduct(const double u[3], const double v[3]) {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

inline double DotProduct(const float u[3], const double v[3]) {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

inline double DotProductWithMax(const float u[3], const double v[3]) {
  return (std::max<double>(u[0], u[0] * v[0]) +
          std::max<double>(u[1], u[1] * v[1]) +
          std::max<double>(u[2], u[2] * v[2]));
}

// Computes a horizontal convolution and transposes the result.
static void Convolution(size_t xsize, size_t ysize,
                        size_t xstep,
                        size_t len, size_t offset,
                        const float* __restrict__ multipliers,
                        const float* __restrict__ inp,
                        float* __restrict__ result) {
  for (size_t x = 0, ox = 0; x < xsize; x += xstep, ox++) {
    int minx = x < offset ? 0 : x - offset;
    int maxx = std::min(xsize, x + len - offset) - 1;
    double weight = 0.0;
    for (int j = minx; j <= maxx; ++j) {
      weight += multipliers[j - x + offset];
    }
    double scale = 1.0 / weight;
    for (size_t y = 0; y < ysize; ++y) {
      double sum = 0.0;
      for (int j = minx; j <= maxx; ++j) {
        sum += inp[y * xsize + j] * multipliers[j - x + offset];
      }
      result[ox * ysize + y] = sum * scale;
    }
  }
}

void GaussBlurApproximation(size_t xsize, size_t ysize, float* channel,
                                   double sigma) {
  double m = 2.25;  // Accuracy increases when m is increased.
  const double scaler = -1.0 / (2 * sigma * sigma);
  // For m = 9.0: exp(-scaler * diff * diff) < 2^ {-52}
  const int diff = std::max<int>(1, m * fabs(sigma));
  const int expn_size = 2 * diff + 1;
  std::vector<float> expn(expn_size);
  for (int i = -diff; i <= diff; ++i) {
    expn[i + diff] = exp(scaler * i * i);
  }
  // No effort was expended to choose good values here.
  const int xstep = std::max(1, int(sigma / 3));
  const int ystep = xstep;
  int dxsize = (xsize + xstep - 1) / xstep;
  int dysize = (ysize + ystep - 1) / ystep;
  std::vector<float> tmp(dxsize * ysize);
  std::vector<float> downsampled_output(dxsize * dysize);
  Convolution(xsize, ysize, xstep, expn_size, diff, expn.data(), channel,
              tmp.data());
  Convolution(ysize, dxsize, ystep, expn_size, diff, expn.data(), tmp.data(),
              downsampled_output.data());
  for (int y = 0; y < ysize; y++) {
    for (int x = 0; x < xsize; x++) {
      // TODO: Use correct rounding.
      channel[y * xsize + x] =
          downsampled_output[(y / ystep) * dxsize + (x / xstep)];
    }
  }
}


// Model of the gamma derivative in the human eye.
static double GammaDerivativeRaw(double v) {
  // Derivative of the linear to sRGB translation.
  return (v <= 4.1533262364511305)
      ? 6.34239659083478
      : 0.3509337062449116 * pow(v / 255.0, -0.7171642149318425);
}

struct GammaDerivativeTableEntry {
  double slope;
  double constant;
};

static GammaDerivativeTableEntry *NewGammaDerivativeTable() {
  GammaDerivativeTableEntry *kTable = new GammaDerivativeTableEntry[256];
  double prev = GammaDerivativeRaw(0);
  for (int i = 0; i < 255; ++i) {
    const double next = GammaDerivativeRaw(i + 1);
    const double slope = next - prev;
    const double constant = prev - slope * i;
    kTable[i].slope = slope;
    kTable[i].constant = constant;
    prev = next;
  }
  kTable[255].slope = 0.0;
  kTable[255].constant = prev;
  return kTable;
}

static double GammaDerivativeLut(double v) {
  static const GammaDerivativeTableEntry *const kTable =
      NewGammaDerivativeTable();
  const GammaDerivativeTableEntry &entry =
      kTable[static_cast<int>(std::min(std::max(0.0, v), 255.0))];
  return entry.slope * v + entry.constant;
}

// Contrast sensitivity related weights.
static const double *GetContrastSensitivityMatrix() {
  static double csf8x8[64] = {
    0.462845464,
    1.48675033,
    0.774522722,
    0.656786477,
    0.507984559,
    0.51125,
    0.51125,
    0.55125,
    1.48675033,
    0.893383342,
    0.729597657,
    0.644616012,
    0.47125,
    0.47125,
    0.53125,
    0.53125,
    0.774522722,
    0.729597657,
    0.669401271,
    0.547687084,
    0.47125,
    0.47125,
    0.53125,
    0.53125,
    0.656786477,
    0.644616012,
    0.547687084,
    0.47125,
    0.47125,
    0.47125,
    0.47125,
    0.47125,
    0.507984559,
    0.47125,
    0.47125,
    0.47125,
    0.47125,
    0.47125,
    0.47125,
    0.47125,
    0.51125,
    0.47125,
    0.47125,
    0.47125,
    0.47125,
    0.53125,
    0.53125,
    0.51125,
    0.51125,
    0.53125,
    0.53125,
    0.47125,
    0.47125,
    0.53125,
    0.47125,
    0.47125,
    0.55125,
    0.53125,
    0.53125,
    0.47125,
    0.47125,
    0.51125,
    0.47125,
    0.51125,
  };
  return &csf8x8[0];
}

static void Transpose8x8(double data[64]) {
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < i; j++) {
      std::swap(data[8 * i + j], data[8 * j + i]);
    }
  }
}

// Perform a DCT on each of the 8 columns. Results is scaled.
static void ButteraugliDctd8x8Vertical(double data[64]) {
  const size_t STRIDE = 8;
  for (int col = 0; col < 8; col++) {
    double *dataptr = &data[col];
    double tmp0 = dataptr[STRIDE * 0] + dataptr[STRIDE * 7];
    double tmp7 = dataptr[STRIDE * 0] - dataptr[STRIDE * 7];
    double tmp1 = dataptr[STRIDE * 1] + dataptr[STRIDE * 6];
    double tmp6 = dataptr[STRIDE * 1] - dataptr[STRIDE * 6];
    double tmp2 = dataptr[STRIDE * 2] + dataptr[STRIDE * 5];
    double tmp5 = dataptr[STRIDE * 2] - dataptr[STRIDE * 5];
    double tmp3 = dataptr[STRIDE * 3] + dataptr[STRIDE * 4];
    double tmp4 = dataptr[STRIDE * 3] - dataptr[STRIDE * 4];

    /* Even part */
    double tmp10 = tmp0 + tmp3;  /* phase 2 */
    double tmp13 = tmp0 - tmp3;
    double tmp11 = tmp1 + tmp2;
    double tmp12 = tmp1 - tmp2;

    dataptr[STRIDE * 0] = tmp10 + tmp11;  /* phase 3 */
    dataptr[STRIDE * 4] = tmp10 - tmp11;

    double z1 = (tmp12 + tmp13) * 0.7071067811865476;  /* c4 */
    dataptr[STRIDE * 2] = tmp13 + z1;  /* phase 5 */
    dataptr[STRIDE * 6] = tmp13 - z1;

    /* Odd part */
    tmp10 = tmp4 + tmp5;  /* phase 2 */
    tmp11 = tmp5 + tmp6;
    tmp12 = tmp6 + tmp7;

    double z5 = (tmp10 - tmp12) * 0.38268343236508984;  /* c6 */
    double z2 = 0.5411961001461969 * tmp10 + z5;  /* c2-c6 */
    double z4 = 1.3065629648763766 * tmp12 + z5;  /* c2+c6 */
    double z3 = tmp11 * 0.7071067811865476;  /* c4 */

    double z11 = tmp7 + z3;  /* phase 5 */
    double z13 = tmp7 - z3;

    dataptr[STRIDE * 5] = z13 + z2;  /* phase 6 */
    dataptr[STRIDE * 3] = z13 - z2;
    dataptr[STRIDE * 1] = z11 + z4;
    dataptr[STRIDE * 7] = z11 - z4;
  }
}

void ButteraugliDctd8x8(double m[64]) {
  static const double kScalingFactors[64] = {
    0.0156250000000000, 0.0079655559235025,
    0.0084561890647843, 0.0093960138583601,
    0.0110485434560398, 0.0140621284865065,
    0.0204150463261934, 0.0400455538709610,
    0.0079655559235025, 0.0040608051949085,
    0.0043109278012960, 0.0047900463261934,
    0.0056324986094293, 0.0071688109352157,
    0.0104075003643000, 0.0204150463261934,
    0.0084561890647843, 0.0043109278012960,
    0.0045764565439602, 0.0050850860570641,
    0.0059794286307045, 0.0076103690966521,
    0.0110485434560398, 0.0216724975831585,
    0.0093960138583601, 0.0047900463261934,
    0.0050850860570641, 0.0056502448912957,
    0.0066439851153692, 0.0084561890647843,
    0.0122764837247985, 0.0240811890647843,
    0.0110485434560398, 0.0056324986094293,
    0.0059794286307045, 0.0066439851153692,
    0.0078125000000000, 0.0099434264107253,
    0.0144356176954889, 0.0283164826985277,
    0.0140621284865065, 0.0071688109352157,
    0.0076103690966521, 0.0084561890647843,
    0.0099434264107253, 0.0126555812845451,
    0.0183730562878025, 0.0360400463261934,
    0.0204150463261934, 0.0104075003643000,
    0.0110485434560398, 0.0122764837247985,
    0.0144356176954889, 0.0183730562878025,
    0.0266735434560398, 0.0523220375957595,
    0.0400455538709610, 0.0204150463261934,
    0.0216724975831585, 0.0240811890647843,
    0.0283164826985277, 0.0360400463261934,
    0.0523220375957595, 0.1026333686292507,
  };
  ButteraugliDctd8x8Vertical(m);
  Transpose8x8(m);
  ButteraugliDctd8x8Vertical(m);
  for (size_t i = 0; i < 64; i++) {
    m[i] *= kScalingFactors[i];
  }
}

// Mix a little bit of neighbouring pixels into the corners.
void ButteraugliMixCorners(double m[64]) {
  static const double c = 5.772211696386835;
  const double w = 1.0 / (c + 2);
  m[0] = (c * m[0] + m[1] + m[8]) * w;
  m[7] = (c * m[7] + m[6] + m[15]) * w;
  m[56] = (c * m[56] + m[57] + m[48]) * w;
  m[63] = (c * m[63] + m[55] + m[62]) * w;
}

// Computes 8x8 DCT (with corner mixing) of each channel of rgb0 and rgb1 and
// adds the total squared 3-dimensional rgbdiff of the two blocks to diff_xyz.
void ButteraugliDctd8x8RgbDiff(const double gamma[3],
                               double rgb0[192],
                               double rgb1[192],
                               double diff_xyz_dc[3],
                               double diff_xyz_ac[3]) {
  const double *csf8x8 = GetContrastSensitivityMatrix();
  for (int c = 0; c < 3; ++c) {
    ButteraugliMixCorners(&rgb0[c * 64]);
    ButteraugliDctd8x8(&rgb0[c * 64]);
    ButteraugliMixCorners(&rgb1[c * 64]);
    ButteraugliDctd8x8(&rgb1[c * 64]);
  }
  double *r0 = &rgb0[0];
  double *g0 = &rgb0[64];
  double *b0 = &rgb0[2 * 64];
  double *r1 = &rgb1[0];
  double *g1 = &rgb1[64];
  double *b1 = &rgb1[2 * 64];
  double rmul = gamma[0];
  double gmul = gamma[1];
  double bmul = gamma[2];
  RgbDiffLowFreqSquaredXyzAccumulate(rmul * (r0[0] - r1[0]),
                                     gmul * (g0[0] - g1[0]),
                                     bmul * (b0[0] - b1[0]),
                                     0, 0, 0, csf8x8[0] * csf8x8[0],
                                     diff_xyz_dc);
  for (size_t i = 1; i < 64; ++i) {
    double d = csf8x8[i] * csf8x8[i];
    RgbDiffSquaredXyzAccumulate(
        rmul * r0[i], gmul * g0[i], bmul * b0[i],
        rmul * r1[i], gmul * g1[i], bmul * b1[i],
        d, diff_xyz_ac);
  }
}

// Direct model with low frequency edge detectors.
// Two edge detectors are applied in each corner of the 8x8 square.
// The squared 3-dimensional error vector is added to diff_xyz.
void Butteraugli8x8CornerEdgeDetectorDiff(
    const size_t pos_x,
    const size_t pos_y,
    const size_t xsize,
    const size_t ysize,
    const std::vector<std::vector<float> > &blurred0,
    const std::vector<std::vector<float> > &blurred1,
    const double gamma[3],
    double diff_xyz[3]) {
  double w = 0.9375862313610259;
  double gamma_scaled[3] = { gamma[0] * w, gamma[1] * w, gamma[2] * w };
  for (int k = 0; k < 4; ++k) {
    double weight = 0.04051114418675643;
    size_t step = 3;
    size_t offset[4][2] = { { 0, 0 }, { 0, 7 }, { 7, 0 }, { 7, 7 } };
    size_t x = pos_x + offset[k][0];
    size_t y = pos_y + offset[k][1];
    if (x >= step && x + step < xsize) {
      size_t ix = y * xsize + (x - step);
      size_t ix2 = ix + 2 * step;
      RgbDiffLowFreqSquaredXyzAccumulate(
          gamma_scaled[0] * (blurred0[0][ix] - blurred0[0][ix2]),
          gamma_scaled[1] * (blurred0[1][ix] - blurred0[1][ix2]),
          gamma_scaled[2] * (blurred0[2][ix] - blurred0[2][ix2]),
          gamma_scaled[0] * (blurred1[0][ix] - blurred1[0][ix2]),
          gamma_scaled[1] * (blurred1[1][ix] - blurred1[1][ix2]),
          gamma_scaled[2] * (blurred1[2][ix] - blurred1[2][ix2]),
          weight, diff_xyz);
    }
    if (y >= step && y + step < ysize) {
      size_t ix = (y - step) * xsize + x ;
      size_t ix2 = ix + 2 * step * xsize;
      RgbDiffLowFreqSquaredXyzAccumulate(
          gamma_scaled[0] * (blurred0[0][ix] - blurred0[0][ix2]),
          gamma_scaled[1] * (blurred0[1][ix] - blurred0[1][ix2]),
          gamma_scaled[2] * (blurred0[2][ix] - blurred0[2][ix2]),
          gamma_scaled[0] * (blurred1[0][ix] - blurred1[0][ix2]),
          gamma_scaled[1] * (blurred1[1][ix] - blurred1[1][ix2]),
          gamma_scaled[2] * (blurred1[2][ix] - blurred1[2][ix2]),
          weight, diff_xyz);
    }
  }
}

double GammaDerivativeWeightedAvg(const double m0[64],
                                  const double m1[64]) {
  double total = 0.0;
  double total_level = 0.0;
  static double slack = 32;
  for (int i = 0; i < 64; ++i) {
    double level = std::min(m0[i], m1[i]);
    double w = GammaDerivativeLut(level) * fabs(m0[i] - m1[i]);
    w += slack;
    total += w;
    total_level += w * level;
  }
  return GammaDerivativeLut(total_level / total);
}

void MixGamma(double gamma[3]) {
  const double gamma_r = gamma[0];
  const double gamma_g = gamma[1];
  const double gamma_b = gamma[2];
  {
    const double a1 = 0.031357286184508865;
    const double a0 = 1.0700981209942968 - a1;
    gamma[0] = pow(gamma_r, a0) * pow(gamma_g, a1);
  }
  {
    const double a0 = 0.022412459763909404;
    const double a1 = 1.0444144247574045 - a0;
    gamma[1] = pow(gamma_r, a0) * pow(gamma_g, a1);
  }
  {
    const double a0 = 0.021865296182264373;
    const double a1 = 0.040708035758485354;
    const double a2 = 0.9595217356556563 - a0 - a1;
    gamma[2] = pow(gamma_r, a0) * pow(gamma_g, a1) * pow(gamma_b, a2);
  }
}

ButteraugliComparator::ButteraugliComparator(
    size_t xsize, size_t ysize, int step)
    : xsize_(xsize),
      ysize_(ysize),
      num_pixels_(xsize * ysize),
      step_(step),
      res_xsize_((xsize + step - 1) / step),
      res_ysize_((ysize + step - 1) / step),
      scale_xyz_(3, std::vector<float>(num_pixels_)),
      gamma_map_(3 * res_xsize_ * res_ysize_),
      dct8x8map_dc_(3 * res_xsize_ * res_ysize_),
      dct8x8map_ac_(3 * res_xsize_ * res_ysize_),
      edge_detector_map_(3 * res_xsize_ * res_ysize_) {}

void ButteraugliComparator::DistanceMap(
    const std::vector<std::vector<float> > &rgb0,
    const std::vector<std::vector<float> > &rgb1,
    std::vector<float> &result) {
  std::vector<bool> changed(res_xsize_ * res_ysize_, true);
  blurred0_.clear();
  DistanceMapIncremental(rgb0, rgb1, changed, result);
}

void ButteraugliComparator::DistanceMapIncremental(
    const std::vector<std::vector<float> > &rgb0,
    const std::vector<std::vector<float> > &rgb1,
    const std::vector<bool>& changed,
    std::vector<float> &result) {
  assert(8 <= xsize_);
  for (int i = 0; i < 3; i++) {
    assert(rgb0[i].size() == num_pixels_);
    assert(rgb1[i].size() == num_pixels_);
  }
  Dct8x8mapIncremental(rgb0, rgb1, changed);
  EdgeDetectorMap(rgb0, rgb1);
  SuppressionMap(rgb0, rgb1);
  FinalizeDistanceMap(&result);
}

void ButteraugliComparator::Dct8x8mapIncremental(
    const std::vector<std::vector<float> > &rgb0,
    const std::vector<std::vector<float> > &rgb1,
    const std::vector<bool>& changed) {
  for (size_t res_y = 0; res_y + 7 < ysize_; res_y += step_) {
    for (size_t res_x = 0; res_x + 7 < xsize_; res_x += step_) {
      size_t res_ix = (res_y * res_xsize_ + res_x) / step_;
      if (!changed[res_ix]) continue;
      double block0[3 * 64];
      double block1[3 * 64];
      double gamma[3];
      for (int i = 0; i < 3; ++i) {
        double* m0 = &block0[i * 64];
        double* m1 = &block1[i * 64];
        for (size_t y = 0; y < 8; y++) {
          for (size_t x = 0; x < 8; x++) {
            m0[8 * y + x] = rgb0[i][(res_y + y) * xsize_ + res_x + x];
            m1[8 * y + x] = rgb1[i][(res_y + y) * xsize_ + res_x + x];
          }
        }
        gamma[i] = GammaDerivativeWeightedAvg(m0, m1);
      }
      MixGamma(gamma);
      double diff_xyz_dc[3] = { 0.0 };
      double diff_xyz_ac[3] = { 0.0 };
      ButteraugliDctd8x8RgbDiff(gamma, block0, block1,
                                diff_xyz_dc, diff_xyz_ac);
      for (int i = 0; i < 3; ++i) {
        gamma_map_[3 * res_ix + i] = gamma[i];
        dct8x8map_dc_[3 * res_ix + i] = diff_xyz_dc[i];
        dct8x8map_ac_[3 * res_ix + i] = diff_xyz_ac[i];
      }
    }
  }
}

void ButteraugliComparator::EdgeDetectorMap(
    const std::vector<std::vector<float> > &rgb0,
    const std::vector<std::vector<float> > &rgb1) {
  static const double kSigma[3] = {
    1.9467950320244936,
    1.9844023344574777,
    0.9443734014996432,
  };
  if (blurred0_.empty()) {
    blurred0_ = rgb0;
    for (int i = 0; i < 3; i++) {
      GaussBlurApproximation(xsize_, ysize_, blurred0_[i].data(), kSigma[i]);
    }
  }
  std::vector<std::vector<float> > blurred1(rgb1);
  for (int i = 0; i < 3; i++) {
    GaussBlurApproximation(xsize_, ysize_, blurred1[i].data(), kSigma[i]);
  }
  for (size_t res_y = 0; res_y + 7 < ysize_; res_y += step_) {
    for (size_t res_x = 0; res_x + 7 < xsize_; res_x += step_) {
      size_t res_ix = (res_y * res_xsize_ + res_x) / step_;
      double gamma[3];
      for (int i = 0; i < 3; ++i) {
        gamma[i] = gamma_map_[3 * res_ix + i];
      }
      double diff_xyz[3] = { 0.0 };
      Butteraugli8x8CornerEdgeDetectorDiff(res_x, res_y, xsize_, ysize_,
                                           blurred0_, blurred1,
                                           gamma, diff_xyz);
      for (int i = 0; i < 3; ++i) {
        edge_detector_map_[3 * res_ix + i] = diff_xyz[i];
      }
    }
  }
}

void ButteraugliComparator::CombineChannels(std::vector<float>* result) {
  result->resize(res_xsize_ * res_ysize_);
  for (size_t res_y = 0; res_y + 7 < ysize_; res_y += step_) {
    for (size_t res_x = 0; res_x + 7 < xsize_; res_x += step_) {
      size_t res_ix = (res_y * res_xsize_ + res_x) / step_;
      double scale[3];
      for (int i = 0; i < 3; ++i) {
        scale[i] = scale_xyz_[i][(res_y + 3) * xsize_ + (res_x + 3)];
        scale[i] *= scale[i];
      }
      (*result)[res_ix] =
          // Apply less suppression to the low frequency component, otherwise
          // it will not rate a difference on a noisy image where the average
          // color changes visibly (especially for blue) high enough. The "max"
          // formula is chosen ad-hoc, as a balance between fixing the noisy
          // image issue (the ideal fix would mean only multiplying the diff
          // with a constant here), and staying aligned with user ratings on
          // other images: take the highest diff of both options.
          (DotProductWithMax(&dct8x8map_dc_[3 * res_ix], scale) +
           DotProduct(&dct8x8map_ac_[3 * res_ix], scale) +
           DotProduct(&edge_detector_map_[3 * res_ix], scale));
    }
  }
}

static double SoftClampHighValues(double v) {
  if (v < 0) {
    return 0;
  }
  return log(v + 1.0);
}

// Making a cluster of local errors to be more impactful than
// just a single error.
void ApplyErrorClustering(const size_t xsize, const size_t ysize,
                          const int step,
                          std::vector<float>* distmap) {
  // Upsample and take square root.
  std::vector<float> distmap_out(xsize * ysize);
  const size_t res_xsize = (xsize + step - 1) / step;
  for (size_t res_y = 0; res_y + 7 < ysize; res_y += step) {
    for (size_t res_x = 0; res_x + 7 < xsize; res_x += step) {
      size_t res_ix = (res_y * res_xsize + res_x) / step;
      double val = sqrt((*distmap)[res_ix]);
      for (size_t off_y = 0; off_y < step; ++off_y) {
        for (size_t off_x = 0; off_x < step; ++off_x) {
          distmap_out[(res_y + off_y) * xsize + res_x + off_x] = val;
        }
      }
    }
  }
  *distmap = distmap_out;
  {
    static const double kSigma = 17.77401417750591;
    static const double mul1 = 6.127295867565043;
    static const double mul2 = 1.5735505590879941;
    std::vector<float> blurred(*distmap);
    GaussBlurApproximation(xsize, ysize, blurred.data(), kSigma);
    for (size_t i = 0; i < ysize * xsize; ++i) {
      (*distmap)[i] += mul1 * SoftClampHighValues(mul2 * blurred[i]);
    }
  }
  {
    static const double kSigma = 11.797090536094919;
    static const double mul1 = 2.9304661498619393;
    static const double mul2 = 2.5400911895122853;
    std::vector<float> blurred(*distmap);
    GaussBlurApproximation(xsize, ysize, blurred.data(), kSigma);
    for (size_t i = 0; i < ysize * xsize; ++i) {
      (*distmap)[i] += mul1 * SoftClampHighValues(mul2 * blurred[i]);
    }
  }
}

static void MultiplyScalarImage(
    size_t xsize, size_t ysize, size_t offset,
    const std::vector<float> &scale, std::vector<float> *result) {
  for (size_t y = 0; y < ysize; ++y) {
    for (size_t x = 0; x < xsize; ++x) {
      size_t idx = std::min<size_t>(y + offset, ysize - 1) * xsize;
      idx += std::min<size_t>(x + offset, xsize - 1);
      double v = scale[idx];
      assert(0 < v);
      (*result)[x + y * xsize] *= v;
    }
  }
}

static void ScaleImage(double scale, std::vector<float> *result) {
  for (size_t i = 0; i < result->size(); ++i) {
    (*result)[i] *= scale;
  }
}

void ButteraugliMap(
    size_t xsize, size_t ysize,
    const std::vector<std::vector<float> > &rgb0,
    const std::vector<std::vector<float> > &rgb1,
    std::vector<float> &result) {
  ButteraugliComparator butteraugli(xsize, ysize, 3);
  butteraugli.DistanceMap(rgb0, rgb1, result);
}

void ButteraugliComparator::SuppressionMap(
    const std::vector<std::vector<float> > &rgb0,
    const std::vector<std::vector<float> > &rgb1) {
  std::vector<std::vector<float> > rgb_avg(3);
  for (int i = 0; i < 3; i++) {
    rgb_avg[i].resize(num_pixels_);
    for (size_t x = 0; x < num_pixels_; ++x) {
      rgb_avg[i][x] = (rgb0[i][x] + rgb1[i][x]) * 0.5;
    }
  }
  SuppressionRgb(rgb_avg, xsize_, ysize_, &scale_xyz_);
}

void ButteraugliComparator::FinalizeDistanceMap(
    std::vector<float>* result) {
  CombineChannels(result);
  ApplyErrorClustering(xsize_, ysize_, step_, result);
  ScaleImage(kGlobalScale, result);
}

double ButteraugliDistanceFromMap(
    size_t xsize, size_t ysize,
    const std::vector<float>& distmap) {
  double retval = 0.0;
  for (size_t y = 0; y + 7 < ysize; ++y) {
    for (size_t x = 0; x + 7 < xsize; ++x) {
      double v = distmap[y * xsize + x];
      if (v > retval) {
        retval = v;
      }
    }
  }
  return retval;
}

const double *GetHighFreqColorDiffDx() {
  static const double kHighFrequencyColorDiffDx[21] = {
    0,
    0.2907966745057564,
    0.4051314804057564,
    0.4772760965057564,
    0.5051493856057564,
    2.1729859604144055,
    3.3884646055698626,
    4.0229515574578265,
    4.816434992428891,
    4.902122343469863,
    5.340254095828891,
    5.575275366425944,
    6.2865515546259445,
    6.8836782708259445,
    7.441068346525944,
    7.939018098125944,
    8.369172080625944,
    8.985082806466515,
    9.334801499366515,
    9.589734180466515,
    9.810687724466515,
  };
  return &kHighFrequencyColorDiffDx[0];
};

const double *GetLowFreqColorDiff() {
  static const double kLowFrequencyColorDiff[21] = {
    0,
    1.1,
    1.5876101261422835,
    1.6976101261422836,
    1.7976101261422834,
    1.8476101261422835,
    1.8976101261422835,
    4.397240606610438,
    4.686599568139378,
    6.186599568139378,
    6.486599568139378,
    6.686599568139378,
    6.886599568139378,
    7.086599568139378,
    7.229062896585699,
    7.3290628965856985,
    7.429062896585698,
    7.529062896585699,
    7.629062896585698,
    7.729062896585699,
    7.8290628965856985,
  };
  return &kLowFrequencyColorDiff[0];
};


const double *GetHighFreqColorDiffDy() {
  static const double kHighFrequencyColorDiffDy[21] = {
    0,
    2.711686498564926,
    6.372713948578674,
    6.749065994565258,
    7.206288924241256,
    7.810763383046433,
    9.06633982465914,
    10.247027693354854,
    11.405609673354855,
    12.514884802454853,
    13.598623651254854,
    14.763543981054854,
    15.858015071354854,
    16.940461051054857,
    18.044423211343567,
    19.148278633543566,
    20.262330053243566,
    21.402794191043565,
    22.499732267943568,
    23.613819632843565,
    24.717384172243566,
  };
  return &kHighFrequencyColorDiffDy[0];
}

const double *GetHighFreqColorDiffDz() {
  static const double kHighFrequencyColorDiffDz[21] = {
    0,
    0.5238354062,
    0.6113836358,
    0.7872517703,
    0.8472517703,
    0.9772517703,
    1.1072517703,
    1.2372517703,
    1.3672517703,
    1.4972517703,
    1.5919694734,
    1.7005177031,
    1.8374517703,
    2.0024517703,
    2.2024517703,
    2.4024517703,
    2.6024517703,
    2.8654517703,
    3.0654517703,
    3.2654517703,
    3.4654517703,
  };
  return &kHighFrequencyColorDiffDz[0];
}

inline double Interpolate(const double *array, int size, double sx) {
  double ix = fabs(sx);
  assert(ix < 10000);
  int baseix = static_cast<int>(ix);
  double res;
  if (baseix >= size - 1) {
    res = array[size - 1];
  } else {
    double mix = ix - baseix;
    int nextix = baseix + 1;
    res = array[baseix] + mix * (array[nextix] - array[baseix]);
  }
  if (sx < 0) res = -res;
  return res;
}

static inline void XyzToVals(double x, double y, double z,
                             double *valx, double *valy, double *valz) {
  static const double xmul = 0.6111808709773186;
  static const double ymul = 0.6254434332781222;
  static const double zmul = 1.3392224065403562;
  *valx = xmul * Interpolate(GetHighFreqColorDiffDx(), 21, x);
  *valy = ymul * Interpolate(GetHighFreqColorDiffDy(), 21, y);
  *valz = zmul * Interpolate(GetHighFreqColorDiffDz(), 21, z);
}

static inline void RgbToVals(double r, double g, double b,
                             double *valx, double *valy, double *valz) {
  double x, y, z;
  RgbToXyz(r, g, b, &x, &y, &z);
  XyzToVals(x, y, z, valx, valy, valz);
}

// Rough psychovisual distance to gray for low frequency colors.
static void RgbLowFreqToVals(double r, double g, double b,
                             double *valx, double *valy, double *valz) {
  static const double mul0 = 1.174114936496674;
  static const double mul1 = 1.1447743969198858;
  static const double a0 = mul0 * 0.1426666666666667;
  static const double a1 = mul0 * -0.065;
  static const double b0 = mul1 * 0.10;
  static const double b1 = mul1 * 0.12;
  static const double b2 = mul1 * 0.023721063454977084;
  static const double c0 = 0.14367553580758044;

  double x = a0 * r + a1 * g;
  double y = b0 * r + b1 * g + b2 * b;
  double z = c0 * b;
  static double xmul = 1.0175474206944557;
  static double ymul = 1.0017393502266154;
  static double zmul = 1.0378355409050648;
  *valx = xmul * Interpolate(GetLowFreqColorDiff(), 21, x);
  *valy = ymul * Interpolate(GetHighFreqColorDiffDy(), 21, y);
  // We use the same table for x and z for the low frequency colors.
  *valz = zmul * Interpolate(GetLowFreqColorDiff(), 21, z);
}

void RgbDiffSquaredXyzAccumulate(double r0, double g0, double b0,
                                 double r1, double g1, double b1,
                                 double factor, double res[3]) {
  double valx0, valy0, valz0;
  double valx1, valy1, valz1;
  if (r0 == r1 && g0 == g1 && b0 == b1) {
    return;
  }
  RgbToVals(r0, g0, b0, &valx0, &valy0, &valz0);
  if (r1 == 0.0 && g1 == 0.0 && b1 == 0.0) {
    res[0] += factor * valx0 * valx0;
    res[1] += factor * valy0 * valy0;
    res[2] += factor * valz0 * valz0;
    return;
  }
  RgbToVals(r1, g1, b1, &valx1, &valy1, &valz1);
  // Approximate the distance of the colors by their respective distances
  // to gray.
  double valx = valx0 - valx1;
  double valy = valy0 - valy1;
  double valz = valz0 - valz1;
  res[0] += factor * valx * valx;
  res[1] += factor * valy * valy;
  res[2] += factor * valz * valz;
}

// Function to estimate the psychovisual impact of a high frequency difference.
double RgbDiffSquared(double r0, double g0, double b0,
                      double r1, double g1, double b1) {
  double vals[3] = { 0 };
  RgbDiffSquaredXyzAccumulate(r0, g0, b0, r1, g1, b1, 1.0, vals);
  return vals[0] + vals[1] + vals[2];
}

// Function to estimate the psychovisual impact of a high frequency difference.
double RgbDiffScaledSquared(double r0, double g0, double b0,
                            double r1, double g1, double b1,
                            const double scale[3]) {
  double vals[3] = { 0 };
  RgbDiffSquaredXyzAccumulate(r0, g0, b0, r1, g1, b1, 1.0, vals);
  return DotProduct(vals, scale);
}

double RgbDiff(double r0, double g0, double b0,
               double r1, double g1, double b1) {
  return sqrt(RgbDiffSquared(r0, g0, b0, r1, g1, b1));
}

void RgbDiffLowFreqSquaredXyzAccumulate(double r0, double g0, double b0,
                                        double r1, double g1, double b1,
                                        double factor, double res[3]) {
  double valx0, valy0, valz0;
  double valx1, valy1, valz1;
  RgbLowFreqToVals(r0, g0, b0, &valx0, &valy0, &valz0);
  if (r1 == 0.0 && g1 == 0.0 && b1 == 0.0) {
    res[0] += factor * valx0 * valx0;
    res[1] += factor * valy0 * valy0;
    res[2] += factor * valz0 * valz0;
    return;
  }
  RgbLowFreqToVals(r1, g1, b1, &valx1, &valy1, &valz1);
  // Approximate the distance of the colors by their respective distances
  // to gray.
  double valx = valx0 - valx1;
  double valy = valy0 - valy1;
  double valz = valz0 - valz1;
  res[0] += factor * valx * valx;
  res[1] += factor * valy * valy;
  res[2] += factor * valz * valz;
}

double RgbDiffLowFreqSquared(double r0, double g0, double b0,
                             double r1, double g1, double b1) {
  double vals[3] = { 0 };
  RgbDiffLowFreqSquaredXyzAccumulate(r0, g0, b0, r1, g1, b1, 1.0, vals);
  return vals[0] + vals[1] + vals[2];
}

double RgbDiffLowFreqScaledSquared(double r0, double g0, double b0,
                                   double r1, double g1, double b1,
                                   const double scale[3]) {
  double vals[3] = { 0 };
  RgbDiffLowFreqSquaredXyzAccumulate(r0, g0, b0, r1, g1, b1, 1.0, vals);
  return DotProduct(vals, scale);
}

double RgbDiffLowFreq(double r0, double g0, double b0,
                      double r1, double g1, double b1) {
  return sqrt(RgbDiffLowFreqSquared(r0, g0, b0, r1, g1, b1));
}

double RgbDiffGamma(double ave_r, double ave_g, double ave_b,
                    double r0, double g0, double b0,
                    double r1, double g1, double b1) {
  const double rmul = GammaDerivativeLut(ave_r);
  const double gmul = GammaDerivativeLut(ave_g);
  const double bmul = GammaDerivativeLut(ave_b);
  return RgbDiff(r0 * rmul, g0 * gmul, b0 * bmul,
                 r1 * rmul, g1 * gmul, b1 * bmul);
}

double RgbDiffGammaLowFreq(double ave_r, double ave_g, double ave_b,
                           double r0, double g0, double b0,
                           double r1, double g1, double b1) {
  const double rmul = GammaDerivativeLut(ave_r);
  const double gmul = GammaDerivativeLut(ave_g);
  const double bmul = GammaDerivativeLut(ave_b);
  return RgbDiffLowFreq(r0 * rmul, g0 * gmul, b0 * bmul,
                        r1 * rmul, g1 * gmul, b1 * bmul);
}

void ButteraugliQuadraticBlockDiffCoeffsXyz(const double scale[3],
                                            const double gamma[3],
                                            const double rgb[192],
                                            double coeffs[192]) {
  double rgb_copy[192];
  memcpy(rgb_copy, rgb, sizeof(rgb_copy));
  for (int c = 0; c < 3; ++c) {
    ButteraugliDctd8x8(&rgb_copy[c * 64]);
  }
  memset(coeffs, 0, 192 * sizeof(coeffs[0]));
  const double *csf8x8 = GetContrastSensitivityMatrix();
  for (int i = 1; i < 64; ++i) {
    double r = gamma[0] * rgb_copy[i];
    double g = gamma[1] * rgb_copy[i + 64];
    double b = gamma[2] * rgb_copy[i + 128];
    double x, y, z;
    RgbToXyz(r, g, b, &x, &y, &z);
    double vals_a[3];
    double vals_b[3];
    XyzToVals(x + 0.5, y + 0.5, z + 0.5, &vals_a[0], &vals_a[1], &vals_a[2]);
    XyzToVals(x - 0.5, y - 0.5, z - 0.5, &vals_b[0], &vals_b[1], &vals_b[2]);
    double slopex = vals_a[0] - vals_b[0];
    double slopey = vals_a[1] - vals_b[1];
    double slopez = vals_a[2] - vals_b[2];
    double d = csf8x8[i] * csf8x8[i];
    coeffs[i] = d * slopex * slopex * scale[0];
    coeffs[i + 64] = d * slopey * slopey * scale[1];
    coeffs[i + 128] = d * slopez * slopez * scale[2];
  }
}

double SuppressionRedPlusGreen(double delta) {
  static double lut[] = {
    2.4080920167439297,
    1.7310517871734234,
    1.6530012641923442,
    1.2793750898946559,
    1.174310066587132,
    1.08674227374109,
    0.9395922737410899,
    0.74560027001709,
    0.625061913513,
    0.585615549987,
    0.545015549987,
    0.523015549987,
    0.523015549987,
  };
  return Interpolate(lut, sizeof(lut) / sizeof(lut[0]), delta);
}

double SuppressionRedMinusGreen(double delta) {
  static double lut[] = {
    3.439854711245826,
    0.9057815912437459,
    0.8795942436511097,
    0.8691824600776474,
    0.8591824600776476,
    0.847448339843269,
    0.82587,
    0.80442724547356859,
    0.69762724547356858,
    0.66102724547356861,
    0.59512224547356851,
    0.57312224547356849,
    0.55312224547356847,
    0.52912224547356856,
    0.50512224547356854,
    0.50512224547356854,
  };
  return Interpolate(lut, sizeof(lut) / sizeof(lut[0]), delta);
}

double SuppressionBlue(double delta) {
  static double lut[] = {
    1.796130974060199,
    1.7586413079453862,
    1.7268200670818195,
    1.6193338330644527,
    1.4578801627801556,
    1.05,
    0.95,
    0.8963665327,
    0.7844709485,
    0.71381616456428487,
    0.67745725036428484,
    0.64597852966428482,
    0.63454542736428488,
    0.6257514661642849,
    0.59191965086428489,
    0.56379229756428484,
    0.53215685696428483,
    0.50415685696428492,
    0.50415685696428492,
  };
  return Interpolate(lut, sizeof(lut) / sizeof(lut[0]), delta);
}

// Replaces values[x + y * xsize] with the minimum of the values in the
// square_size square with coordinates
//   x - offset .. x + square_size - offset - 1,
//   y - offset .. y + square_size - offset - 1.
void MinSquareVal(size_t square_size, size_t offset,
                  size_t xsize, size_t ysize,
                  float *values) {
  // offset is not negative and smaller than square_size.
  assert(offset < square_size);
  std::vector<float> tmp(xsize * ysize);
  for (size_t y = 0; y < ysize; ++y) {
    const size_t minh = offset > y ? 0 : y - offset;
    const size_t maxh = std::min<size_t>(ysize, y + square_size - offset);
    for (size_t x = 0; x < xsize; ++x) {
      double min = values[x + minh * xsize];
      for (size_t j = minh + 1; j < maxh; ++j) {
        min = fmin(min, values[x + j * xsize]);
      }
      tmp[x + y * xsize] = min;
    }
  }
  for (size_t x = 0; x < xsize; ++x) {
    const size_t minw = offset > x ? 0 : x - offset;
    const size_t maxw = std::min<size_t>(xsize, x + square_size - offset);
    for (size_t y = 0; y < ysize; ++y) {
      double min = tmp[minw + y * xsize];
      for (size_t j = minw + 1; j < maxw; ++j) {
        min = fmin(min, tmp[j + y * xsize]);
      }
      values[x + y * xsize] = min;
    }
  }
}

static const int kRadialWeightSize = 5;

double RadialWeights(double* output) {
  const double limit = 1.946968773063937;
  const double range = 0.28838147488925875;
  double total_weight = 0.0;
  for(int i=0;i<kRadialWeightSize;i++) {
    for(int j=0;j<kRadialWeightSize;j++) {
      int ddx = i - kRadialWeightSize / 2;
      int ddy = j - kRadialWeightSize / 2;
      double d = sqrt(ddx*ddx + ddy*ddy);
      double result;
      if (d < limit) {
        result = 1.0;
      } else if (d < limit + range) {
        result = 1.0 - (d - limit) * (1.0 / range);
      } else {
        result = 0;
      }
      output[i*kRadialWeightSize+j] = result;
      total_weight += result;
    }
  }
  return total_weight;
}

// ===== Functions used by Suppression() only =====
void Average5x5(int xsize, int ysize, std::vector<float>* diffs) {
  std::vector<float> tmp = *diffs;
  assert(kRadialWeightSize % 2 == 1);
  double patch[kRadialWeightSize*kRadialWeightSize];
  double total_weight = RadialWeights(patch);
  double total_weight_inv = 1.0 / total_weight;
  for (int y = 0; y < ysize; y++) {
    for (int x = 0; x < xsize; x++) {
      double sum = 0.0;
      for(int dy = 0; dy < kRadialWeightSize; dy++) {
        int yy = y - kRadialWeightSize/2 + dy;
        if (yy < 0 || yy >= ysize) {
          continue;
        }
        int dx = 0;
        const int xlim = std::min(xsize, x + kRadialWeightSize / 2 + 1);
        for (int xx = std::max(0, x - kRadialWeightSize / 2);
             xx < xlim; xx++, dx++) {
          const int ix = yy * xsize + xx;
          double w = patch[dy * kRadialWeightSize + dx];
          sum += w * tmp[ix];
        }
      }
      (*diffs)[y * xsize + x] = sum * total_weight_inv;
    }
  }
}

void DiffPrecompute(
    const std::vector<std::vector<float> > &rgb, size_t xsize, size_t ysize,
    std::vector<std::vector<float> > *suppression) {
  const size_t size = xsize * ysize;
  static const double kSigma[3] = {
    1.5406666666666667,
    1.5745555555555555,
    0.7178888888888888,
  };
  *suppression = rgb;
  const double muls[3] = {
    0.9594825346868103,
    0.9594825346868103,
    0.563781684306615,
  };
  for (int i = 0; i < 3; ++i) {
    GaussBlurApproximation(xsize, ysize, (*suppression)[i].data(), kSigma[i]);
    for (size_t x = 0; x < size; ++x) {
      (*suppression)[i][x] = muls[i] * (rgb[i][x] + (*suppression)[i][x]);
    }
  }
  for (size_t y = 0; y < ysize; ++y) {
    for (size_t x = 0; x < xsize; ++x) {
      size_t ix = x + xsize * y;
      double valsh[3] = { 0.0 };
      double valsv[3] = { 0.0 };
      if (x + 1 < xsize) {
        const int ix2 = ix + 1;
        double ave_r = ((*suppression)[0][ix] + (*suppression)[0][ix2]) * 0.5;
        double ave_g = ((*suppression)[1][ix] + (*suppression)[1][ix2]) * 0.5;
        double ave_b = ((*suppression)[2][ix] + (*suppression)[2][ix2]) * 0.5;
        double gamma[3] = {
          GammaDerivativeLut(ave_r),
          GammaDerivativeLut(ave_g),
          GammaDerivativeLut(ave_b),
        };
        MixGamma(gamma);
        double r0 = gamma[0] * ((*suppression)[0][ix] - (*suppression)[0][ix2]);
        double g0 = gamma[1] * ((*suppression)[1][ix] - (*suppression)[1][ix2]);
        double b0 = gamma[2] * ((*suppression)[2][ix] - (*suppression)[2][ix2]);
        RgbToVals(r0, g0, b0, &valsh[0], &valsh[1], &valsh[2]);
      }
      if (y + 1 < ysize) {
        const int ix2 = ix + xsize;
        double ave_r = ((*suppression)[0][ix] + (*suppression)[0][ix2]) * 0.5;
        double ave_g = ((*suppression)[1][ix] + (*suppression)[1][ix2]) * 0.5;
        double ave_b = ((*suppression)[2][ix] + (*suppression)[2][ix2]) * 0.5;
        double gamma[3] = {
          GammaDerivativeLut(ave_r),
          GammaDerivativeLut(ave_g),
          GammaDerivativeLut(ave_b),
        };
        MixGamma(gamma);
        double r0 = gamma[0] * ((*suppression)[0][ix] - (*suppression)[0][ix2]);
        double g0 = gamma[1] * ((*suppression)[1][ix] - (*suppression)[1][ix2]);
        double b0 = gamma[2] * ((*suppression)[2][ix] - (*suppression)[2][ix2]);
        RgbToVals(r0, g0, b0, &valsv[0], &valsv[1], &valsv[2]);
      }
      for (int i = 0; i < 3; ++i) {
        (*suppression)[i][ix] = 0.5 * (fabs(valsh[i]) + fabs(valsv[i]));
      }
    }
  }
}

void SuppressionRgb(const std::vector<std::vector<float> > &rgb,
                    size_t xsize, size_t ysize,
                    std::vector<std::vector<float> > *suppression) {
  DiffPrecompute(rgb, xsize, ysize, suppression);
  for (int i = 0; i < 3; ++i) {
    Average5x5(xsize, ysize, &(*suppression)[i]);
    MinSquareVal(14, 3, xsize, ysize, (*suppression)[i].data());
    static const double sigma[3] = {
      13.963188902126857,
      14.912114324178102,
      12.316604481444129,
    };
    GaussBlurApproximation(xsize, ysize, (*suppression)[i].data(), sigma[i]);
  }
  for (size_t y = 0; y < ysize; ++y) {
    for (size_t x = 0; x < xsize; ++x) {
      const double muls[3] = {
        34.702156451753055,
        4.259296809697752,
        30.51708015595755,
      };
      const size_t idx = y * xsize + x;
      const double a = (*suppression)[0][idx];
      const double b = (*suppression)[1][idx];
      const double c = (*suppression)[2][idx];
      const double mix = 0.003513839880391094;
      (*suppression)[0][idx] = SuppressionRedMinusGreen(muls[0] * a + mix * b);
      (*suppression)[1][idx] = SuppressionRedPlusGreen(muls[1] * b);
      (*suppression)[2][idx] = SuppressionBlue(muls[2] * c + mix * b);
    }
  }
}

bool ButteraugliInterface(size_t xsize, size_t ysize,
                          const std::vector<std::vector<float> > &rgb0,
                          const std::vector<std::vector<float> > &rgb1,
                          std::vector<float> &diffmap,
                          double &diffvalue) {
  if (xsize < 32 || ysize < 32) {
    return false;  // Butteraugli is undefined for small images.
  }
  size_t size = xsize;
  size *= ysize;
  for (int i = 0; i < 3; i++) {
    if (rgb0[i].size() != size || rgb1[i].size() != size) {
      return false;
    }
  }
  ButteraugliMap(xsize, ysize, rgb0, rgb1, diffmap);
  diffvalue = ButteraugliDistanceFromMap(xsize, ysize, diffmap);
  return true;
}

bool ButteraugliAdaptiveQuantization(size_t xsize, size_t ysize,
    const std::vector<std::vector<float> > &rgb, std::vector<float> &quant) {
  if (xsize < 32 || ysize < 32) {
    return false;  // Butteraugli is undefined for small images.
  }
  size_t size = xsize * ysize;

  std::vector<std::vector<float> > scale_xyz(3);
  SuppressionRgb(rgb, xsize, ysize, &scale_xyz);
  quant.resize(size);

  // Multiply the result of suppression and intensity masking together.
  // Suppression gives us values in 3 color channels, but for now we take only
  // the intensity channel.
  for (size_t i = 0; i < size; i++) {
    quant[i] = scale_xyz[1][i];
  }
  return true;
}

double ButteraugliFuzzyClass(double score) {
  // Interesting values of fuzzy_width range from 10 to 1000. The smaller the
  // value, the smoother the class boundaries, and more images will
  // participate in a higher level optimization.
  static const double fuzzy_width = 55;
  static const double fuzzy_good = fuzzy_width / kButteraugliGood;
  const double good_class =
      1.0 / (1.0 + exp((score - kButteraugliGood) * fuzzy_good));
  static const double fuzzy_ok = fuzzy_width / kButteraugliBad;
  const double ok_class =
      1.0 / (1.0 + exp((score - kButteraugliBad) * fuzzy_ok));
  return ok_class + good_class;
}

}  // namespace butteraugli
