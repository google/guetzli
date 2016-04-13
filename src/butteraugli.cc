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

static const double kInternalGoodQualityThreshold = 14.878153265541;
static const double kGlobalScale = 1.0 / kInternalGoodQualityThreshold;

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

static void GaussBlurApproximation(size_t xsize, size_t ysize, float* channel,
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

#ifdef REVERSE_SRGB
// If we wanted to reverse the sRGB, we could use the code here.

static double GammaDerivativeRaw(double v) {
  // Derivative of the linear to sRGB translation.
  return v <= 255.0 * 0.0031308
             ? 12.92
             : 1.055 / 2.4 * std::pow(v / 255.0, 1.0 / 2.4 - 1.0);
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
    kTable[i] = {slope, constant};
    prev = next;
  }
  kTable[255] = {0.0, prev};
  return kTable;
}

static double GammaDerivativeLut(double v) {
  static const GammaDerivativeTableEntry *const kTable =
      NewGammaDerivativeTable();
  const GammaDerivativeTableEntry &entry =
      kTable[static_cast<int>(std::min(std::max(0.0, v), 255.0))];
  return entry.slope * v + entry.constant;
}
#else
static double GammaDerivativeLut(double v) {
  static const double lut[256] = {
    // Generated with python:
    // >>> ''.join(["6.25, "] + ["%.6f, " % min(6.25, (0.4166 * (i / 255.) ** (-(2.4 - 1.0) / 2.2))) for i in range(1, 256)])
    6.25, 6.250000, 6.250000, 6.250000, 5.861717, 5.085748, 4.528629, 4.105483,
    3.771033, 3.498716, 3.271827, 3.079282, 2.913414, 2.768732, 2.641190, 2.527739,
    2.426028, 2.334216, 2.250838, 2.174711, 2.104872, 2.040524, 1.981002, 1.925750,
    1.874294, 1.826231, 1.781215, 1.738946, 1.699164, 1.661640, 1.626176, 1.592596,
    1.560742, 1.530477, 1.501677, 1.474230, 1.448037, 1.423008, 1.399062, 1.376126,
    1.354133, 1.333021, 1.312735, 1.293225, 1.274443, 1.256347, 1.238897, 1.222058,
    1.205794, 1.190076, 1.174874, 1.160161, 1.145913, 1.132107, 1.118720, 1.105733,
    1.093127, 1.080884, 1.068987, 1.057421, 1.046172, 1.035225, 1.024569, 1.014189,
    1.004076, 0.994218, 0.984606, 0.975228, 0.966077, 0.957144, 0.948420, 0.939897,
    0.931569, 0.923428, 0.915467, 0.907681, 0.900062, 0.892606, 0.885307, 0.878159,
    0.871157, 0.864298, 0.857576, 0.850986, 0.844525, 0.838189, 0.831973, 0.825875,
    0.819891, 0.814016, 0.808249, 0.802585, 0.797023, 0.791558, 0.786189, 0.780913,
    0.775726, 0.770628, 0.765614, 0.760684, 0.755834, 0.751064, 0.746369, 0.741750,
    0.737203, 0.732728, 0.728321, 0.723982, 0.719709, 0.715500, 0.711354, 0.707269,
    0.703244, 0.699277, 0.695368, 0.691514, 0.687714, 0.683968, 0.680274, 0.676630,
    0.673036, 0.669491, 0.665994, 0.662543, 0.659138, 0.655778, 0.652461, 0.649187,
    0.645955, 0.642764, 0.639613, 0.636502, 0.633429, 0.630394, 0.627396, 0.624435,
    0.621509, 0.618618, 0.615762, 0.612939, 0.610149, 0.607392, 0.604666, 0.601972,
    0.599309, 0.596675, 0.594071, 0.591496, 0.588950, 0.586431, 0.583940, 0.581477,
    0.579039, 0.576628, 0.574242, 0.571882, 0.569546, 0.567235, 0.564948, 0.562684,
    0.560444, 0.558226, 0.556031, 0.553858, 0.551706, 0.549576, 0.547467, 0.545378,
    0.543310, 0.541262, 0.539234, 0.537225, 0.535236, 0.533265, 0.531312, 0.529378,
    0.527462, 0.525564, 0.523683, 0.521819, 0.519973, 0.518143, 0.516329, 0.514532,
    0.512751, 0.510985, 0.509235, 0.507501, 0.505781, 0.504076, 0.502387, 0.500711,
    0.499050, 0.497403, 0.495770, 0.494150, 0.492545, 0.490952, 0.489373, 0.487806,
    0.486253, 0.484712, 0.483184, 0.481668, 0.480164, 0.478672, 0.477192, 0.475724,
    0.474267, 0.472821, 0.471387, 0.469965, 0.468553, 0.467152, 0.465761, 0.464381,
    0.463012, 0.461653, 0.460305, 0.458966, 0.457637, 0.456318, 0.455009, 0.453710,
    0.452420, 0.451139, 0.449868, 0.448606, 0.447353, 0.446108, 0.444873, 0.443647,
    0.442429, 0.441220, 0.440019, 0.438826, 0.437642, 0.436466, 0.435298, 0.434138,
    0.432986, 0.431842, 0.430706, 0.429577, 0.428456, 0.427342, 0.426236, 0.425137,
    0.424045, 0.422960, 0.421883, 0.420813, 0.419749, 0.418693, 0.417643, 0.416600,
  };
  if (v < 0) {
    return lut[0];
  }
  int iv = int(v);
  if (iv >= 254) {
    return lut[255];
  }
  int frac = v - iv;
  return frac * lut[iv + 1] + (1.0 - frac) * lut[iv];
}
#endif

// Contrast sensitivity related weights.
static const double *GetContrastSensitivityMatrix() {
  static const double csf8x8[64] = {
    0.4513821, 1.1849009, 0.9597666, 0.7657001, 0.5460047, 0.4145545, 0.3576890, 0.3400602,
    1.1849009, 0.9956394, 0.8497080, 0.6374688, 0.4996464, 0.3667089, 0.3561453, 0.3142601,
    0.9597666, 0.8497080, 0.7195796, 0.5646769, 0.3977972, 0.3698285, 0.3339006, 0.3120368,
    0.7657001, 0.6374688, 0.5646769, 0.4669175, 0.3766784, 0.3570152, 0.3392428, 0.3176755,
    0.5460047, 0.4996464, 0.3977972, 0.3766784, 0.3540639, 0.3518468, 0.3224825, 0.3042312,
    0.4145545, 0.3667089, 0.3698285, 0.3570152, 0.3518468, 0.3475624, 0.3262351, 0.3072478,
    0.3576890, 0.3561453, 0.3339006, 0.3392428, 0.3224825, 0.3262351, 0.3032862, 0.3026028,
    0.3400602, 0.3142601, 0.3120368, 0.3176755, 0.3042312, 0.3072478, 0.3026028, 0.3008635,
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
  static const double c = 5.739760676622537;
  const double w = 1.0 / (c + 2);
  m[0] = (c * m[0] + m[1] + m[8]) * w;
  m[7] = (c * m[7] + m[6] + m[15]) * w;
  m[56] = (c * m[56] + m[57] + m[48]) * w;
  m[63] = (c * m[63] + m[55] + m[62]) * w;
}

// Computes 8x8 DCT (with corner mixing) of each channel of rgb0 and rgb1 and
// returns the total scaled and squared rgbdiff of the two blocks.
double ButteraugliDctd8x8RgbDiff(const double scale[3],
                                 const double gamma[3],
                                 double rgb0[192],
                                 double rgb1[192]) {
  const double *csf8x8 = GetContrastSensitivityMatrix();
  const double fabs_sum_norm = 0.052081605258397466;
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
  double diff = 0;
  diff += csf8x8[0] * csf8x8[0] *
      RgbDiffLowFreqScaledSquared(rmul * (r0[0] - r1[0]),
                                  gmul * (g0[0] - g1[0]),
                                  bmul * (b0[0] - b1[0]),
                                  0, 0, 0, &scale[0]);
  double r_fabs_sum = 0;
  double g_fabs_sum = 0;
  double b_fabs_sum = 0;
  for (size_t i = 1; i < 64; ++i) {
    double d = csf8x8[i] * csf8x8[i];
    diff += d * RgbDiffScaledSquared(
        rmul * r0[i], gmul * g0[i], bmul * b0[i],
        rmul * r1[i], gmul * g1[i], bmul * b1[i],
        &scale[0]);
    if (i >= 24 || (i & 0x7) >= 3) {
      r_fabs_sum += csf8x8[i] * (fabs(r0[i]) - fabs(r1[i]));
      g_fabs_sum += csf8x8[i] * (fabs(g0[i]) - fabs(g1[i]));
      b_fabs_sum += csf8x8[i] * (fabs(b0[i]) - fabs(b1[i]));
    }
  }
  r_fabs_sum *= fabs_sum_norm;
  g_fabs_sum *= fabs_sum_norm;
  b_fabs_sum *= fabs_sum_norm;
  diff += RgbDiffScaledSquared(rmul * r_fabs_sum,
                               gmul * g_fabs_sum,
                               bmul * b_fabs_sum,
                               0, 0, 0, &scale[0]);
  return diff;
}

// Direct model with low frequency edge detectors.
// Two edge detectors are applied in each corner of the 8x8 square.
double Butteraugli8x8CornerEdgeDetectorDiff(
    const size_t pos_x,
    const size_t pos_y,
    const size_t xsize,
    const size_t ysize,
    const std::vector<std::vector<float> > &blurred0,
    const std::vector<std::vector<float> > &blurred1,
    const double scale[3],
    const double gamma[3]) {
  double w = 0.913775;
  double gamma_scaled[3] = { gamma[0] * w, gamma[1] * w, gamma[2] * w };
  double diff = 0.0;
  for (int k = 0; k < 4; ++k) {
    double weight = 0.041709991396540254;
    size_t step = 3;
    size_t offset[4][2] = { { 0, 0 }, { 0, 7 }, { 7, 0 }, { 7, 7 } };
    size_t x = pos_x + offset[k][0];
    size_t y = pos_y + offset[k][1];
    if (x >= step && x + step < xsize) {
      size_t ix = y * xsize + (x - step);
      size_t ix2 = ix + 2 * step;
      diff += weight * RgbDiffLowFreqScaledSquared(
          gamma_scaled[0] * (blurred0[0][ix] - blurred0[0][ix2]),
          gamma_scaled[1] * (blurred0[1][ix] - blurred0[1][ix2]),
          gamma_scaled[2] * (blurred0[2][ix] - blurred0[2][ix2]),
          gamma_scaled[0] * (blurred1[0][ix] - blurred1[0][ix2]),
          gamma_scaled[1] * (blurred1[1][ix] - blurred1[1][ix2]),
          gamma_scaled[2] * (blurred1[2][ix] - blurred1[2][ix2]),
          &scale[0]);
    }
    if (y >= step && y + step < ysize) {
      size_t ix = (y - step) * xsize + x ;
      size_t ix2 = ix + 2 * step * xsize;
      diff += weight * RgbDiffLowFreqScaledSquared(
          gamma_scaled[0] * (blurred0[0][ix] - blurred0[0][ix2]),
          gamma_scaled[1] * (blurred0[1][ix] - blurred0[1][ix2]),
          gamma_scaled[2] * (blurred0[2][ix] - blurred0[2][ix2]),
          gamma_scaled[0] * (blurred1[0][ix] - blurred1[0][ix2]),
          gamma_scaled[1] * (blurred1[1][ix] - blurred1[1][ix2]),
          gamma_scaled[2] * (blurred1[2][ix] - blurred1[2][ix2]),
          &scale[0]);
    }
  }
  return diff;
}

static const double Average(const double m[64]) {
  double r = 0.0;
  for (int i = 0; i < 64; ++i) r += m[i];
  return r * (1.0 / 64.0);
}

double GammaDerivativeAvgMin(const double m0[64], const double m1[64]) {
  return GammaDerivativeLut(std::min(Average(m0), Average(m1)));
}

// weight can be NULL if no weights are used
// suppression can be NULL if no suppression is used.
// result[y*xsize + x] is set for 0 <= x < xsize - 7 and 0 <= y < ysize - 7.
// The value depends on rgb0[i*xsize + j] for x <= i < x + 8 and y <= j < y + 8
// and suppression[i * xsize + j] if suppression != NULL.
void Dctd8x8mapWithRgbDiff(
    const std::vector<std::vector<float> > &rgb0,
    const std::vector<std::vector<float> > &rgb1,
    const std::vector<std::vector<float> > &scale_xyz,
    size_t xsize, size_t ysize,
    std::vector<float> *result) {
  assert(8 <= xsize);
  static const double kSigma[3] = {
    1.9513615245943847,
    2.015905890589516,
    0.9280480893114391,
  };
  std::vector<std::vector<float> > blurred0(rgb0);
  std::vector<std::vector<float> > blurred1(rgb1);
  for (int i = 0; i < 3; i++) {
    GaussBlurApproximation(xsize, ysize, blurred0[i].data(), kSigma[i]);
    GaussBlurApproximation(xsize, ysize, blurred1[i].data(), kSigma[i]);
  }
  const int step = 3;
  for (size_t res_y = 0; res_y + 7 < ysize; res_y += step) {
    for (size_t res_x = 0; res_x + 7 < xsize; res_x += step) {
      double scale[3];
      double block0[3 * 64];
      double block1[3 * 64];
      double gamma[3];
      for (int i = 0; i < 3; ++i) {
        scale[i] = scale_xyz[i][res_y * xsize + res_x];
        scale[i] *= scale[i];
        double* m0 = &block0[i * 64];
        double* m1 = &block1[i * 64];
        for (size_t y = 0; y < 8; y++) {
          for (size_t x = 0; x < 8; x++) {
            m0[8 * y + x] = rgb0[i][(res_y + y) * xsize + res_x + x];
            m1[8 * y + x] = rgb1[i][(res_y + y) * xsize + res_x + x];
          }
        }
        gamma[i] = GammaDerivativeAvgMin(m0, m1);
      }
      double diff =
          ButteraugliDctd8x8RgbDiff(scale, gamma, block0, block1) +
          Butteraugli8x8CornerEdgeDetectorDiff(res_x, res_y, xsize, ysize,
                                               blurred0, blurred1,
                                               scale, gamma);
      diff = sqrt(diff);
      for (size_t off_y = 0; off_y < step; ++off_y) {
        for (size_t off_x = 0; off_x < step; ++off_x) {
          (*result)[(res_y + off_y) * xsize + res_x + off_x] = diff;
        }
      }
    }
  }
}

// Making a cluster of local errors to be more impactful than
// just a single error.
void ApplyErrorClustering(const size_t xsize, const size_t ysize,
                          std::vector<float>* distmap) {
  {
    static const double kSigma = 18.09238864420438;
    static const double mul1 = 5.0;
    std::vector<float> blurred(*distmap);
    GaussBlurApproximation(xsize, ysize, blurred.data(), kSigma);
    for (size_t i = 0; i < ysize * xsize; ++i) {
      (*distmap)[i] += mul1 * sqrt(blurred[i]);
    }
  }
  {
    static const double kSigma = 13.952277501928073;
    static const double mul1 = 5.0;
    std::vector<float> blurred(*distmap);
    GaussBlurApproximation(xsize, ysize, blurred.data(), kSigma);
    for (size_t i = 0; i < ysize * xsize; ++i) {
      (*distmap)[i] += mul1 * sqrt(blurred[i]);
    }
  }
  {
    static const double kSigma = 0.8481105317429303;
    GaussBlurApproximation(xsize, ysize, distmap->data(), kSigma);
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
  const size_t size = xsize * ysize;
  for (int i = 0; i < 3; i++) {
    assert(rgb0[i].size() == size);
    assert(rgb1[i].size() == size);
  }
  std::vector<std::vector<float> > rgb_avg(3);
  std::vector<std::vector<float> > scale_xyz(3);
  for (int i = 0; i < 3; i++) {
    scale_xyz[i].resize(size);
    rgb_avg[i].resize(size);
    for (size_t x = 0; x < size; ++x) {
      rgb_avg[i][x] = (rgb0[i][x] + rgb1[i][x]) * 0.5;
    }
  }
  result.resize(size);
  SuppressionRgb(rgb_avg, xsize, ysize, &scale_xyz);
  Dctd8x8mapWithRgbDiff(rgb0, rgb1, scale_xyz, xsize, ysize, &result);
  ApplyErrorClustering(xsize, ysize, &result);
  ScaleImage(kGlobalScale, &result);
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
    0.2553299613,
    0.3696647672,
    0.4418093833,
    0.4696826724,
    1.2712216081,
    1.7641138989,
    2.5116164028,
    3.2777716368,
    4.5607862469,
    5.0846053503,
    5.8894534019,
    6.6007295901,
    7.1978563063,
    7.755246382,
    8.2531961336,
    8.6833501161,
    9.0542933365,
    9.4040120294,
    9.6589447105,
    9.8798982545,
  };
  return &kHighFrequencyColorDiffDx[0];
};

const double *GetLowFreqColorDiff() {
  static const double kLowFrequencyColorDiff[21] = {
    0,
    0.3,
    0.79,
    0.9,
    1.0,
    1.05,
    1.1,
    3.4,
    4.9,
    6.4,
    6.7,
    6.9,
    7.1,
    7.3,
    7.4,
    7.5,
    7.6,
    7.7,
    7.8,
    7.9,
    8.0,
  };
  return &kLowFrequencyColorDiff[0];
};


const double *GetHighFreqColorDiffDy() {
  static const double kHighFrequencyColorDiffDy[21] = {
    0,
    2.5055555555555555,
    5.025128048592593,
    6.0451851851851846,
    6.9919222806,
    8.031863896,
    9.1596687733,
    10.2235384948,
    11.3821204748,
    12.4913956039,
    13.5751344527,
    14.7400547825,
    15.8345258728,
    16.9169718525,
    18.0305120521,
    19.1343674743,
    20.248418894,
    21.3888830318,
    22.4858211087,
    23.5999084736,
    24.703473013,
  };
  return &kHighFrequencyColorDiffDy[0];
}

const double *GetHighFreqColorDiffDz() {
  static const double kHighFrequencyColorDiffDz[21] = {
    0,
    0.5240354062,
    0.6115836358,
    0.7874517703,
    0.8474517703,
    0.9774517703,
    1.1074517703,
    1.2374517703,
    1.3674517703,
    1.4974517703,
    1.6519694734,
    1.7795177031,
    1.9174517703,
    2.0874517703,
    2.2874517703,
    2.4874517703,
    2.6874517703,
    2.8874517703,
    3.0874517703,
    3.2874517703,
    3.4874517703,
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
  static const double xmul = 0.39901253189500346;
  static const double ymul = 0.6048133735736677;
  static const double zmul = 1.2517479414904453;
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
  static const double mul0 = 1.0990671172402453;
  static const double mul1 = 1.1274953108568997;
  static const double a0 = mul0 * 0.1426666666666667;
  static const double a1 = mul0 * -0.065;
  static const double b0 = mul1 * 0.10;
  static const double b1 = mul1 * 0.12;
  static const double b2 = mul1 * 0.02340234375;
  static const double c0 = 0.14846361693942411;

  double x = a0 * r + a1 * g;
  double y = b0 * r + b1 * g + b2 * b;
  double z = c0 * b;
  static double xmul = 0.7670979081732033;
  static double ymul = 0.9488415893585428;
  static double zmul = 1.010766219034261;
  *valx = xmul * Interpolate(GetLowFreqColorDiff(), 21, x);
  *valy = ymul * Interpolate(GetHighFreqColorDiffDy(), 21, y);
  // We use the same table for x and z for the low frequency colors.
  *valz = zmul * Interpolate(GetLowFreqColorDiff(), 21, z);
}

// Function to estimate the psychovisual impact of a high frequency difference.
double RgbDiffSquared(double r0, double g0, double b0,
                      double r1, double g1, double b1) {
  double valx0, valy0, valz0;
  double valx1, valy1, valz1;
  if (r0 == r1 && g0 == g1 && b0 == b1) {
    return 0.0;
  }
  RgbToVals(r0, g0, b0, &valx0, &valy0, &valz0);
  if (r1 == 0.0 && g1 == 0.0 && b1 == 0.0) {
    return valx0 * valx0 + valy0 * valy0 + valz0 * valz0;
  }
  RgbToVals(r1, g1, b1, &valx1, &valy1, &valz1);
  // Approximate the distance of the colors by their respective distances
  // to gray.
  double valx = valx0 - valx1;
  double valy = valy0 - valy1;
  double valz = valz0 - valz1;
  return valx * valx + valy * valy + valz * valz;
}

// Function to estimate the psychovisual impact of a high frequency difference.
double RgbDiffScaledSquared(double r0, double g0, double b0,
                            double r1, double g1, double b1,
                            const double scale[3]) {
  double valx0, valy0, valz0;
  double valx1, valy1, valz1;
  if (r0 == r1 && g0 == g1 && b0 == b1) {
    return 0.0;
  }
  RgbToVals(r0, g0, b0, &valx0, &valy0, &valz0);
  if (r1 == 0.0 && g1 == 0.0 && b1 == 0.0) {
    return valx0 * valx0 * scale[0] + valy0 * valy0 * scale[1]
        + valz0 * valz0 * scale[2];
  }
  RgbToVals(r1, g1, b1, &valx1, &valy1, &valz1);
  // Approximate the distance of the colors by their respective distances
  // to gray.
  double valx = valx0 - valx1;
  double valy = valy0 - valy1;
  double valz = valz0 - valz1;
  return valx * valx * scale[0] + valy * valy * scale[1]
      + valz * valz * scale[2];
}

void RgbDiffSquaredMultiChannel(double r0, double g0, double b0, double *diff) {
  double valx, valy, valz;
  RgbToVals(r0, g0, b0, &valx, &valy, &valz);
  diff[0] = valx * valx;
  diff[1] = valy * valy;
  diff[2] = valz * valz;
}

double RgbDiff(double r0, double g0, double b0,
               double r1, double g1, double b1) {
  return sqrt(RgbDiffSquared(r0, g0, b0, r1, g1, b1));
}

double RgbDiffLowFreqSquared(double r0, double g0, double b0,
                             double r1, double g1, double b1) {
  double valx0, valy0, valz0;
  double valx1, valy1, valz1;
  RgbLowFreqToVals(r0, g0, b0, &valx0, &valy0, &valz0);
  if (r1 == 0.0 && g1 == 0.0 && b1 == 0.0) {
    return valx0 * valx0 + valy0 * valy0 + valz0 * valz0;
  }
  RgbLowFreqToVals(r1, g1, b1, &valx1, &valy1, &valz1);
  // Approximate the distance of the colors by their respective distances
  // to gray.
  double valx = valx0 - valx1;
  double valy = valy0 - valy1;
  double valz = valz0 - valz1;
  return valx * valx + valy * valy + valz * valz;
}

double RgbDiffLowFreqScaledSquared(double r0, double g0, double b0,
                                   double r1, double g1, double b1,
                                   const double scale[3]) {
  double valx0, valy0, valz0;
  double valx1, valy1, valz1;
  RgbLowFreqToVals(r0, g0, b0, &valx0, &valy0, &valz0);
  if (r1 == 0.0 && g1 == 0.0 && b1 == 0.0) {
    return valx0 * valx0 * scale[0] + valy0 * valy0 * scale[1]
           + valz0 * valz0 * scale[2];
  }
  RgbLowFreqToVals(r1, g1, b1, &valx1, &valy1, &valz1);
  // Approximate the distance of the colors by their respective distances
  // to gray.
  double valx = valx0 - valx1;
  double valy = valy0 - valy1;
  double valz = valz0 - valz1;
  return valx * valx * scale[0] + valy * valy * scale[1]
      + valz * valz * scale[2];
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
    2.2038456986630592,
    1.5731079166300341,
    1.5025796916608852,
    1.240385560834905,
    1.1080040420826909,
    0.97302,
    0.82587,
    0.7401324462890625,
    0.6623163635253906,
    0.62287,
    0.58227,
    0.56027,
    0.54027,
    0.51627,
    0.49227,
    0.46777,
    0.44327,
    0.41527,
    0.39227,
    0.36727,
    0.34527,
    0.34527,
  };
  return Interpolate(lut, sizeof(lut) / sizeof(lut[0]), delta);
}

double SuppressionRedMinusGreen(double delta) {
  static double lut[] = {
    1.9132731619110555,
    1.1827935095723001,
    1.166906161979664,
    1.14765450496,
    1.0961,
    0.97302,
    0.82587,
    0.729,
    0.6222,
    0.5856,
    0.519695,
    0.497695,
    0.477695,
    0.453695,
    0.429695,
    0.405195,
    0.380695,
    0.352695,
    0.329695,
    0.304695,
    0.282695,
    0.282695,
  };
  return Interpolate(lut, sizeof(lut) / sizeof(lut[0]), delta);
}

double SuppressionBlue(double delta) {
  static double lut[] = {
    1.7831013900451262,
    1.7756117239303137,
    1.733790483066747,
    1.52,
    1.4,
    1.05,
    0.95,
    0.8963665327,
    0.7844709485,
    0.5895605387,
    0.5532016245,
    0.5217229038,
    0.5102898015,
    0.5014958403,
    0.467664025,
    0.4395366717,
    0.4079012311,
    0.3799012311,
    0.3569012311,
    0.3319012311,
    0.3099012311,
    0.3099012311,
  };
  return Interpolate(lut, sizeof(lut) / sizeof(lut[0]), delta);
}

// mins[x + y * xsize] is the minimum of the values in the 14x14 square.
// mins[x + y * xsize] is the minimum
// of the values in the square_size square with coordinates
//   x - offset .. x + square_size - offset - 1,
//   y - offset .. y + square_size - offset - 1.
void MinSquareVal(size_t square_size, size_t offset,
                  size_t xsize, size_t ysize,
                  const float *values,
                  float *mins) {
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
      mins[x + y * xsize] = min;
    }
  }
}

static const int kRadialWeightSize = 5;

void RadialWeights(double* output) {
  const double limit = 1.917;
  const double range = 0.34;
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
    }
  }
}

// ===== Functions used by Suppression() only =====
void Average5x5(const std::vector<float> &localh,
                  const std::vector<float> &localv,
                  int xsize, int ysize, std::vector<float>* output) {
  assert(kRadialWeightSize % 2 == 1);
  double patch[kRadialWeightSize*kRadialWeightSize];
  RadialWeights(patch);
  for (int y = 0; y < ysize; y++) {
    for (int x = 0; x < xsize; x++) {
      double sum = 0.0;
      double total_weight = 0.0;
      for(int dy = 0; dy < kRadialWeightSize; dy++) {
        int yy = y - kRadialWeightSize/2 + dy;
        if (yy < 0 || yy >= ysize) {
          continue;
        }
        for (int xx = std::max(0, x - kRadialWeightSize / 2), dx = 0,
                 xlim = std::min(xsize, x + kRadialWeightSize / 2 + 1);
             xx < xlim; xx++, dx++) {
          const int ix = yy * xsize + xx;
          double w = patch[dy * kRadialWeightSize + dx];
          sum += w * (localh[ix] + localv[ix]);
          total_weight += 2*w;
        }
      }
      (*output)[y * xsize + x] = sum / total_weight;
    }
  }
}

void DiffPrecompute(
    const std::vector<std::vector<float> > &rgb, size_t xsize, size_t ysize,
    std::vector<std::vector<float> > *htab,
    std::vector<std::vector<float> > *vtab) {
  for (int i = 0; i < 3; ++i) {
    (*vtab)[i].resize(xsize * ysize);
    (*htab)[i].resize(xsize * ysize);
  }
  for (size_t y = 0; y < ysize; ++y) {
    for (size_t x = 0; x < xsize; ++x) {
      size_t ix = x + xsize * y;
      if (x + 1 < xsize) {
        const int ix2 = ix + 1;
        double ave_r = (rgb[0][ix] + rgb[0][ix2]) * 0.5;
        double ave_g = (rgb[1][ix] + rgb[1][ix2]) * 0.5;
        double ave_b = (rgb[2][ix] + rgb[2][ix2]) * 0.5;
        double mul_r = GammaDerivativeLut(ave_r);
        double mul_g = GammaDerivativeLut(ave_g);
        double mul_b = GammaDerivativeLut(ave_b);
        double diff[3];
        RgbDiffSquaredMultiChannel(mul_r * (rgb[0][ix] - rgb[0][ix2]),
                                   mul_g * (rgb[1][ix] - rgb[1][ix2]),
                                   mul_b * (rgb[2][ix] - rgb[2][ix2]),
                                   &diff[0]);
        (*htab)[0][ix] = sqrt(diff[0]);
        (*htab)[1][ix] = sqrt(diff[1]);
        (*htab)[2][ix] = sqrt(diff[2]);
      }
      if (y + 1 < ysize) {
        const int ix2 = ix + xsize;
        double ave_r = (rgb[0][ix] + rgb[0][ix2]) * 0.5;
        double ave_g = (rgb[1][ix] + rgb[1][ix2]) * 0.5;
        double ave_b = (rgb[2][ix] + rgb[2][ix2]) * 0.5;
        double mul_r = GammaDerivativeLut(ave_r);
        double mul_g = GammaDerivativeLut(ave_g);
        double mul_b = GammaDerivativeLut(ave_b);
        double diff[3];
        RgbDiffSquaredMultiChannel(mul_r * (rgb[0][ix] - rgb[0][ix2]),
                                   mul_g * (rgb[1][ix] - rgb[1][ix2]),
                                   mul_b * (rgb[2][ix] - rgb[2][ix2]),
                                   &diff[0]);
        (*vtab)[0][ix] = sqrt(diff[0]);
        (*vtab)[1][ix] = sqrt(diff[1]);
        (*vtab)[2][ix] = sqrt(diff[2]);
      }
    }
  }
}

void SuppressionRgb(const std::vector<std::vector<float> > &rgb,
                    size_t xsize, size_t ysize,
                    std::vector<std::vector<float> > *suppression) {
  size_t size = xsize * ysize;
  std::vector<std::vector<float> > localh(3);
  std::vector<std::vector<float> > localv(3);
  std::vector<std::vector<float> > local(3);
  for (int i = 0; i < 3; ++i) {
    (*suppression)[i].resize(size);
    localh[i].resize(size);
    localv[i].resize(size);
    local[i].resize(size);
  }
  static const double kSigma[3] = {
    1.5806666666666667,
    1.6145555555555555,
    0.7578888888888888,
  };
  std::vector<std::vector<float> > rgb_blurred(rgb);
  const double muls[3] = { 0.92,
                           0.92,
                           0.58 };
  for (int i = 0; i < 3; ++i) {
    GaussBlurApproximation(xsize, ysize, rgb_blurred[i].data(), kSigma[i]);
    for (size_t x = 0; x < size; ++x) {
      rgb_blurred[i][x] = muls[i] * (rgb[i][x] + rgb_blurred[i][x]);
    }
  }
  DiffPrecompute(rgb_blurred, xsize, ysize, &localh, &localv);
  for (int i = 0; i < 3; ++i) {
    Average5x5(localh[i], localv[i], xsize, ysize, &local[i]);
    MinSquareVal(14, 3, xsize, ysize,
                 local[i].data(), (*suppression)[i].data());
    static const double sigma[3] = {
      16.442676435716145,
      15.698088616478731,
      9.33141632753124,
    };
    GaussBlurApproximation(xsize, ysize, (*suppression)[i].data(), sigma[i]);
  }
  for (size_t y = 0; y < ysize; ++y) {
    for (size_t x = 0; x < xsize; ++x) {
      const double muls[3] = {
        34.56456168215157,
        4.163065644554275,
        30.499881780067923,
      };
      const size_t idx = y * xsize + x;
      const double a = (*suppression)[0][idx];
      const double b = (*suppression)[1][idx];
      const double c = (*suppression)[2][idx];
      const double mix = 0.0;
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
