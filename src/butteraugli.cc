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

#include <algorithm>

namespace butteraugli {

// Computes a horizontal convolution and transposes the result.
static void Convolution(size_t xsize, size_t ysize,
                        size_t len, size_t offset,
                        const double* __restrict__ multipliers,
                        const double* __restrict__ inp,
                        double* __restrict__ result) {
  for (size_t x = 0; x < xsize; ++x) {
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
      result[x * ysize + y] = sum * scale;
    }
  }
}

static void GaussBlurApproximation(size_t xsize, size_t ysize, double* channel,
                                   double sigma) {
  double m = 4.5;  // Accuracy increases when m is increased.
  const double scaler = -1.0 / (2 * sigma * sigma);
  // For m = 9.0: exp(-scaler * diff * diff) < 2^ {-52}
  const int diff = m * fabs(sigma);
  const int expn_size = 2 * diff + 1;
  std::vector<double> expn(expn_size);
  for (int i = -diff; i <= diff; ++i) {
    expn[i + diff] = exp(scaler * i * i);
  }
  std::vector<double> tmp(xsize * ysize);
  Convolution(xsize, ysize, expn_size, diff, expn.data(), channel, tmp.data());
  Convolution(ysize, xsize, expn_size, diff, expn.data(), tmp.data(), channel);
}

static double GammaDerivativeLut(double v) {
  static const double lut[256] = {
    // Generated with python:
    // >>> ''.join([7.097] + ["%.6f, " % min(7.097, (0.43732443835227586 *
    // (i / 255.) ** (-(2.2 - 1.0) / 2.2))) for i in range(1, 256)])
    7.097, 7.097,
    6.155507, 4.934170, 4.217603, 3.734270,
    3.380772, 3.108132, 2.889798, 2.709980,
    2.558630, 2.429012, 2.316422, 2.217464,
    2.129616, 2.050962, 1.980019, 1.915614,
    1.856812, 1.802852, 1.753111, 1.707071,
    1.664299, 1.624431, 1.587156, 1.552206,
    1.519352, 1.488395, 1.459161, 1.431497,
    1.405269, 1.380359, 1.356660, 1.334079,
    1.312532, 1.291942, 1.272242, 1.253370,
    1.235270, 1.217892, 1.201188, 1.185118,
    1.169643, 1.154727, 1.140337, 1.126444,
    1.113021, 1.100040, 1.087480, 1.075318,
    1.063533, 1.052107, 1.041023, 1.030263,
    1.019812, 1.009656, 0.999781, 0.990175,
    0.980827, 0.971724, 0.962856, 0.954214,
    0.945788, 0.937570, 0.929550, 0.921723,
    0.914079, 0.906612, 0.899315, 0.892182,
    0.885207, 0.878385, 0.871709, 0.865175,
    0.858778, 0.852514, 0.846377, 0.840363,
    0.834469, 0.828691, 0.823025, 0.817467,
    0.812014, 0.806663, 0.801411, 0.796254,
    0.791190, 0.786217, 0.781331, 0.776530,
    0.771812, 0.767174, 0.762614, 0.758131,
    0.753721, 0.749383, 0.745115, 0.740915,
    0.736782, 0.732713, 0.728707, 0.724763,
    0.720878, 0.717052, 0.713283, 0.709570,
    0.705911, 0.702304, 0.698750, 0.695246,
    0.691791, 0.688385, 0.685026, 0.681712,
    0.678444, 0.675220, 0.672038, 0.668899,
    0.665801, 0.662744, 0.659725, 0.656746,
    0.653804, 0.650899, 0.648031, 0.645198,
    0.642400, 0.639636, 0.636905, 0.634207,
    0.631542, 0.628908, 0.626304, 0.623731,
    0.621188, 0.618674, 0.616188, 0.613731,
    0.611301, 0.608898, 0.606522, 0.604172,
    0.601848, 0.599548, 0.597274, 0.595023,
    0.592797, 0.590594, 0.588414, 0.586257,
    0.584121, 0.582008, 0.579917, 0.577846,
    0.575796, 0.573767, 0.571758, 0.569769,
    0.567799, 0.565848, 0.563916, 0.562003,
    0.560108, 0.558231, 0.556372, 0.554530,
    0.552706, 0.550898, 0.549107, 0.547332,
    0.545574, 0.543831, 0.542104, 0.540393,
    0.538697, 0.537015, 0.535349, 0.533697,
    0.532060, 0.530436, 0.528827, 0.527231,
    0.525649, 0.524080, 0.522525, 0.520982,
    0.519453, 0.517936, 0.516431, 0.514939,
    0.513459, 0.511991, 0.510534, 0.509090,
    0.507657, 0.506235, 0.504825, 0.503425,
    0.502037, 0.500659, 0.499292, 0.497936,
    0.496590, 0.495254, 0.493928, 0.492612,
    0.491307, 0.490011, 0.488724, 0.487447,
    0.486180, 0.484922, 0.483673, 0.482433,
    0.481202, 0.479980, 0.478766, 0.477562,
    0.476366, 0.475178, 0.473998, 0.472827,
    0.471664, 0.470510, 0.469363, 0.468224,
    0.467092, 0.465969, 0.464853, 0.463745,
    0.462644, 0.461550, 0.460464, 0.459385,
    0.458313, 0.457248, 0.456191, 0.455140,
    0.454096, 0.453058, 0.452028, 0.451004,
    0.449986, 0.448975, 0.447970, 0.446972,
    0.445980, 0.444994, 0.444015, 0.443041,
    0.442074, 0.441112, 0.440157, 0.439207,
    0.438263, 0.437324,
  };
  if (v < 0) {
    return 0;
  }
  if (v >= 255) {
    return lut[255];
  }
  int iv = int(v + 0.5);
  return lut[iv];
}

// Contrast sensitivity related weights.
static const double csf8x8[64] = {
  0.43, 1.00, 0.96, 0.77, 0.55, 0.38, 0.32, 0.32,
  1.00, 0.95, 0.85, 0.64, 0.50, 0.32, 0.31, 0.29,
  0.96, 0.85, 0.72, 0.51, 0.34, 0.31, 0.29, 0.29,
  0.77, 0.64, 0.51, 0.40, 0.31, 0.29, 0.29, 0.29,
  0.55, 0.50, 0.34, 0.31, 0.29, 0.29, 0.29, 0.29,
  0.38, 0.32, 0.31, 0.29, 0.29, 0.29, 0.29, 0.29,
  0.32, 0.31, 0.29, 0.29, 0.29, 0.29, 0.29, 0.30,
  0.32, 0.29, 0.29, 0.29, 0.29, 0.29, 0.30, 0.31,
};

static void Transpose8x8(double data[64]) {
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < i; j++) {
      std::swap(data[8 * i + j], data[8 * j + i]);
    }
  }
}

static void ScalingFactors(double data[64]) {
  // The scale of the discrete cosine transform.
  // dctf8x8Scale[i] = 1 if i==0 else 2*cos(pi*i/16)
  const double scale = 1/64.0;
  const double dctd8x8ScaleInverse[8] = {
    1, 0.5097955791041592, 0.541196100146197, 0.6013448869350453,
    0.7071067811865475, 0.8999762231364156, 1.3065629648763764,
    2.5629154477415055};
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      data[8 * i + j] = dctd8x8ScaleInverse[i] * dctd8x8ScaleInverse[j] * scale;
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

// weight can be NULL if no weights are used
// suppression can be NULL if no suppression is used.
// result[y*xsize + x] is set for 0 <= x < xsize - 7 and 0 <= y < ysize - 7.
// The value depends on rgb0[i*xsize + j] for x <= i < x + 8 and y <= j < y + 8
// and suppression[i * xsize + j] if suppression != NULL.
static void Dctd8x8mapWithRgbDiff(
    const std::vector<std::vector<double> > &rgb0,
    const std::vector<std::vector<double> > &rgb1,
    const std::vector<std::vector<double> > &blurred0,
    const std::vector<std::vector<double> > &blurred1,
    const std::vector<std::vector<double> > &scale_xyz,
    size_t xsize, size_t ysize,
    std::vector<double> *result) {
  assert(8 <= xsize);
  std::vector<double> scaling_factors(64);
  ScalingFactors(scaling_factors.data());

  double fabs_sum_norm = 0;
  {
    for (size_t i = 1; i < 64; ++i) {
      if (i >= 24 || (i & 0x7) >= 3) {
        fabs_sum_norm += csf8x8[i];
      }
    }
    fabs_sum_norm = 1.1 / fabs_sum_norm;
  }
  // Works pretty well and faster with 3, requires possibly some more tuning.
  const int step = 1;
  for (size_t res_y = 0; res_y + 7 < ysize; res_y += step) {
    for (size_t res_x = 0; res_x + 7 < xsize; res_x += step) {
      double scale[3];
      for (int i = 0; i < 3; ++i) {
        scale[i] = scale_xyz[i][res_y * xsize + res_x];
        scale[i] *= scale[i];
      }
      double a[6 * 64];
      for (int c = 0; c < 6; ++c) {
        double *m = &a[c * 64];
        const std::vector<double> &channel = c < 3 ? rgb0[c] : rgb1[c - 3];
        for (size_t y = 0; y < 8; y++) {
          for (size_t x = 0; x < 8; x++) {
            m[8 * y + x] = channel[(res_y + y) * xsize + res_x + x];
          }
        }
        {
          // Mix a little bit of neighbouring pixels into the corners.
          const double c = 5.8;
          const double w = 1.0 / (c + 2);
          m[0] = (c * m[0] + m[1] + m[8]) * w;
          m[7] = (c * m[7] + m[6] + m[15]) * w;
          m[56] = (c * m[56] + m[57] + m[48]) * w;
          m[63] = (c * m[63] + m[55] + m[62]) * w;
        }
        ButteraugliDctd8x8Vertical(m);
        Transpose8x8(m);
        ButteraugliDctd8x8Vertical(m);
        for (size_t i = 0; i < 64; i++) {
          m[i] *= scaling_factors[i];
        }
      }
      double *r0 = &a[0];
      double *g0 = &a[64];
      double *b0 = &a[2 * 64];
      double *r1 = &a[3 * 64];
      double *g1 = &a[4 * 64];
      double *b1 = &a[5 * 64];
      double avg_r = std::min(r0[0], r1[0]);
      double avg_g = std::min(g0[0], g1[0]);
      double avg_b = std::min(b0[0], b1[0]);
      double rmul = GammaDerivativeLut(avg_r);
      double gmul = GammaDerivativeLut(avg_g);
      double bmul = GammaDerivativeLut(avg_b);
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
      double w = 0.93;
      rmul *= w;
      gmul *= w;
      bmul *= w;
      for (int k = 0; k < 4; ++k) {
        // Direct model with low frequency edge detectors.
        // Two edge detectors are applied in each corner of the 8x8 square.
        double weight = 0.0255;
        size_t step = 3;
        size_t offset[4][2] = { { 0, 0 }, { 0, 7 }, { 7, 0 }, { 7, 7 } };
        size_t x = res_x + offset[k][0];
        size_t y = res_y + offset[k][1];
        if (x >= step && x + step < xsize) {
          size_t ix = y * xsize + (x - step);
          size_t ix2 = ix + 2 * step;
          diff += weight * RgbDiffLowFreqScaledSquared(
              rmul * (blurred0[0][ix] - blurred0[0][ix2]),
              gmul * (blurred0[1][ix] - blurred0[1][ix2]),
              bmul * (blurred0[2][ix] - blurred0[2][ix2]),
              rmul * (blurred1[0][ix] - blurred1[0][ix2]),
              gmul * (blurred1[1][ix] - blurred1[1][ix2]),
              bmul * (blurred1[2][ix] - blurred1[2][ix2]),
              &scale[0]);
        }
        if (y >= step && y + step < ysize) {
          size_t ix = (y - step) * xsize + x ;
          size_t ix2 = ix + 2 * step * xsize;
          diff += weight * RgbDiffLowFreqScaledSquared(
              rmul * (blurred0[0][ix] - blurred0[0][ix2]),
              gmul * (blurred0[1][ix] - blurred0[1][ix2]),
              bmul * (blurred0[2][ix] - blurred0[2][ix2]),
              rmul * (blurred1[0][ix] - blurred1[0][ix2]),
              gmul * (blurred1[1][ix] - blurred1[1][ix2]),
              bmul * (blurred1[2][ix] - blurred1[2][ix2]),
              &scale[0]);
        }
      }
      (*result)[res_y * xsize + res_x] = sqrt(diff);
    }
  }
  {
    // Shaping the exponentiation/locality curve.
    // This gives possibly about 1-1.5 % more accuracy.
    const double kSigma = 7.0;
    std::vector<double> blurred(*result);
    GaussBlurApproximation(xsize, ysize, blurred.data(), kSigma);
    for (size_t i = 0; i < ysize * xsize; ++i) {
      (*result)[i] += 0.75 * blurred[i] + 1.9 * sqrt(blurred[i]);
    }
  }
}

static void MultiplyScalarImage(
    size_t xsize, size_t ysize, size_t offset,
    const std::vector<double> &scale, std::vector<double> *result) {
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

void ButteraugliMap(
    size_t xsize, size_t ysize,
    const std::vector<std::vector<double> > &rgb0,
    const std::vector<std::vector<double> > &rgb1,
    std::vector<double> &result) {
  const size_t size = xsize * ysize;
  for (int i = 0; i < 3; i++) {
    assert(rgb0[i].size() == size);
    assert(rgb1[i].size() == size);
  }
  std::vector<std::vector<double> > rgb_avg(3);
  const double kSigma[3] = { 1.66, 1.66, 0.75 };
  std::vector<std::vector<double> > blurred0(rgb0);
  std::vector<std::vector<double> > blurred1(rgb1);
  std::vector<std::vector<double> > blurred_avg(3);
  std::vector<std::vector<double> > scale_xyz(3);
  for (int i = 0; i < 3; i++) {
    GaussBlurApproximation(xsize, ysize, blurred0[i].data(), kSigma[i]);
    GaussBlurApproximation(xsize, ysize, blurred1[i].data(), kSigma[i]);
    rgb_avg[i].resize(size);
    blurred_avg[i].resize(size);
    for (size_t x = 0; x < size; ++x) {
      rgb_avg[i][x] = (rgb0[i][x] + rgb1[i][x]) * 0.5;
      blurred_avg[i][x] = (blurred0[i][x] + blurred1[i][x]) * 0.5;
    }
    scale_xyz[i].resize(size);
  }
  result.resize(size);
  SuppressionRgb(rgb_avg, blurred_avg, xsize, ysize, &scale_xyz);
  Dctd8x8mapWithRgbDiff(rgb0, rgb1, blurred0, blurred1,
                        scale_xyz, xsize, ysize, &result);
  std::vector<double> scale2(size);
  IntensityMasking(rgb_avg, xsize, ysize, scale2.data());
  MultiplyScalarImage(xsize, ysize, 4, scale2, &result);
}

double ButteraugliDistanceFromMap(
    size_t xsize, size_t ysize,
    const std::vector<double>& distmap) {
  double p = 16.0;
  double sum = 0.0;
  for (size_t y = 0; y + 7 < ysize; ++y) {
    for (size_t x = 0; x + 7 < xsize; ++x) {
      sum += pow(distmap[y * xsize + x], p);
    }
  }
  int squares = (xsize - 7) * (ysize - 7);
  return pow(sum / squares, 1.0 / p);
}

double IntensityMaskingNonlinearity(double val) {
  static const double lut[] = {
    0.70, 0.90, 0.96, 0.98, 0.99, 0.996, 0.998, 0.999, 1.0
  };
  if (val < 0) {
    if (val < -0.1) {
      printf("negative val\n");
    }
    val = 0;
  }
  int lut_size = sizeof(lut) / sizeof(lut[0]);
  val *= lut_size;
  double lut_pos = val;
  int ix = static_cast<int>(lut_pos);
  double frac = lut_pos - ix;
  double one_minus_frac = 1.0 - frac;
  if (ix < 0) {
    return lut[0];
  }
  if (ix >= lut_size - 1) {
    return lut[lut_size - 1];
  }
  return (frac * lut[ix + 1] + one_minus_frac * lut[ix]);
}

void IntensityMasking(const std::vector<std::vector<double> > &rgb,
                      size_t xsize, size_t ysize,
                      double *mask) {
  size_t size = xsize * ysize;
  std::vector<double> intensity(size);
  for (size_t i = 0; i < size; ++i) {
    double r = rgb[0][i];
    double g = rgb[1][i];
    double b = rgb[2][i];
    intensity[i] = 0.3 * r + 0.59 * g + 0.11 * b;
  }
  std::vector<double> blur(intensity);
  double kSigmaSmooth = 15.0;
  GaussBlurApproximation(xsize, ysize, blur.data(), kSigmaSmooth);
  for (size_t i = 0; i < size; ++i) {
    if (blur[i] >= intensity[i]) {
      mask[i] = IntensityMaskingNonlinearity(intensity[i] / blur[i]);
    } else {
      mask[i] = 1.0;
    }
  }
  std::vector<double> mask_copy(mask, mask + size);
  GaussBlurApproximation(xsize, ysize, mask_copy.data(), 10.0);
  for (size_t i = 0; i < size; ++i) {
    if (mask[i] > mask_copy[i]) {
      mask_copy[i] = mask[i];
    }
  }
  GaussBlurApproximation(xsize, ysize, mask_copy.data(), 1.5);
  for (size_t i = 0; i < size; ++i) {
    mask[i] = mask_copy[i];
  }
}

static const double kHighFrequencyColorDiffDx[21] = {
  0, 0.1232098765, 0.2898765432, 0.5074074074, 0.7590123457, 1.0960493827,
  1.5790123457, 2.2407407407, 3.0654320988, 3.9901234568, 4.9407407407,
  5.8456790123, 6.6592592593, 7.3555555556, 7.9425925926, 8.437037037,
  8.8580246914, 9.212962963, 9.512962963, 9.7740740741, 10.0
};

static const double kHighFrequencyColorDiffDy[21] = {
  0, 2.19, 3.62, 4.945, 6.3086419753, 7.5907407407, 8.7962962963, 9.9567901235,
  11.1037037037, 12.250617284, 13.4, 14.5481481481, 15.6888888889,
  16.8141975309, 17.9222222222, 19.0259259259, 20.149382716, 21.312962963,
  22.5185185185, 23.7530864198, 25,
};

static const double kHighFrequencyColorDiffDz[21] = {
  0.0, 0.06, 0.125, 0.21, 0.42, 0.63, 0.84, 1.05, 1.26, 1.47, 1.68, 1.89, 2.1,
  2.31, 2.52, 2.73, 2.94, 3.15, 3.36, 3.57, 3.78
};

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

static inline void RgbToXyz(double r, double g, double b,
                            double *valx, double *valy, double *valz) {
  *valx = 0.171 * r - 0.0812 * g;
  *valy = 0.08265 * r + 0.168 * g;
  *valz = 0.183333 * b;
}

// Rough psychovisual distance to gray.
static inline void RgbToVals(double r, double g, double b,
                             double *valx, double *valy, double *valz) {
  double x, y, z;
  RgbToXyz(r, g, b, &x, &y, &z);
  *valx = Interpolate(&kHighFrequencyColorDiffDx[0], 21, x);
  *valy = Interpolate(&kHighFrequencyColorDiffDy[0], 21, y);
  *valz = Interpolate(&kHighFrequencyColorDiffDz[0], 21, z);
}

// Rough psychovisual distance to gray for low frequency colors.
static void RgbLowFreqToVals(double r, double g, double b,
                             double *valx, double *valy, double *valz) {
  double x = 0.171 * r - 0.0812 * g;
  double y = 0.08265 * r + 0.168 * g + 0.01283326 * b;
  double z = 0.183333 * b;
  *valx = Interpolate(&kHighFrequencyColorDiffDx[0], 21, x);
  *valy = Interpolate(&kHighFrequencyColorDiffDy[0], 21, y);
  // We use the x-table for z for the low frequency colors.
  *valz = Interpolate(&kHighFrequencyColorDiffDx[0], 21, z);
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

double SuppressionRedPlusGreen(double delta) {
  static const double lut[] = {
    1.465, 1.4425, 1.420, 1.2555, 1.091, 0.9605, 0.830, 0.725, 0.620, 0.5835,
    0.547, 0.524, 0.501, 0.477, 0.453, 0.4285, 0.404, 0.378, 0.352, 0.327,
    0.302,
  };
  if (delta < 0) {
    if (delta < -0.1) {
      printf("negative delta\n");
    }
    delta = 0;
  }
  double lut_pos = delta;
  int ix = static_cast<int>(lut_pos);
  double frac = lut_pos - ix;
  double one_minus_frac = 1.0 - frac;
  int lut_size = sizeof(lut) / sizeof(lut[0]);
  if (ix < 0) {
    return lut[0];
  }
  if (ix >= lut_size - 1) {
    return lut[lut_size - 1];
  }
  return (frac * lut[ix + 1] + one_minus_frac * lut[ix]);
}

double SuppressionRedMinusGreen(double delta) {
  return SuppressionRedPlusGreen(delta);
}

double SuppressionBlue(double delta) {
  return SuppressionRedPlusGreen(delta);
}

// mins[x + y * xsize] is the minimum of the values in the 14x14 square.
// mins[x + y * xsize] is the minimum
// of the values in the square_size square with coordinates
//   x - offset .. x + square_size - offset - 1,
//   y - offset .. y + square_size - offset - 1.
void MinSquareVal(size_t square_size, size_t offset,
                  size_t xsize, size_t ysize,
                  const double *values,
                  double *mins) {
  // offset is not negative and smaller than square_size.
  assert(offset < square_size);
  std::vector<double> tmp(xsize * ysize);
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

double RadialWeight(double d) {
  const double limit = 1.85;
  const double range = 0.54;
  d -= limit;
  if (d < 0.0) {
    return 1;
  }
  if (d >= range) {
    return 0;
  }
  return 1.0 - d * (1.0 / range);
}

// ===== Functions used by Suppression() only =====
double Average5x5(const std::vector<double> &localh,
                  const std::vector<double> &localv,
                  int xsize, int ysize, int x, int y) {
  double retval = 0;
  x -= 1;
  y -= 1;
  double n = 0;
  for (int dy = 0; dy < 7; ++dy) {
    if (y + dy < 0 || y + dy >= ysize) {
      continue;
    }
    for (int dx = 0; dx < 7; ++dx) {
      if (x + dx < 0 || x + dx >= xsize) {
        continue;
      }
      const double ddx = dx - 3;
      const double ddy = dy - 3;
      const int ix = (y + dy) * xsize + x + dx;
      if (x + dx + 1 < xsize) {
        double w = RadialWeight(sqrt((ddx + 0.5) * (ddx + 0.5) + ddy * ddy));
        retval += w * localh[ix];
        n += w;
      }
      if (y + dy + 1 < ysize) {
        double w = RadialWeight(sqrt(ddx * ddx + (ddy + 0.5) * (ddy + 0.5)));
        retval += w * localv[ix];
        n += w;
      }
    }
  }
  retval /= n;
  return retval;
}

void DiffPrecompute(
    const std::vector<std::vector<double> > &rgb, size_t xsize, size_t ysize,
    std::vector<std::vector<double> > *htab,
    std::vector<std::vector<double> > *vtab) {
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

void SuppressionRgb(const std::vector<std::vector<double> > &rgb,
                    const std::vector<std::vector<double> > &blurred,
                    size_t xsize, size_t ysize,
                    std::vector<std::vector<double> > *suppression) {
  size_t size = xsize * ysize;
  std::vector<std::vector<double> > localh(3);
  std::vector<std::vector<double> > localv(3);
  std::vector<std::vector<double> > local(3);
  for (int i = 0; i < 3; ++i) {
    (*suppression)[i].resize(size);
    localh[i].resize(size);
    localv[i].resize(size);
    local[i].resize(size);
  }
  std::vector<std::vector<double> > rgb_blurred(3);
  const double muls[3] = { 0.9135, 0.9135, 0.6825 };
  for (int i = 0; i < 3; ++i) {
    rgb_blurred[i].resize(size);
    for (size_t x = 0; x < size; ++x) {
      rgb_blurred[i][x] = muls[i] * (rgb[i][x] + blurred[i][x]);
    }
  }
  DiffPrecompute(rgb_blurred, xsize, ysize, &localh, &localv);
  for (int i = 0; i < 3; ++i) {
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        size_t idx = y * xsize + x;
        local[i][idx] =
            Average5x5(localh[i], localv[i], xsize, ysize, x, y);
      }
    }
    MinSquareVal(14, 3, xsize, ysize,
                 local[i].data(), (*suppression)[i].data());
    double sigma[3] = { 17.0, 14.95, 17.0 };
    GaussBlurApproximation(xsize, ysize, (*suppression)[i].data(), sigma[i]);
  }
  for (size_t y = 0; y < ysize; ++y) {
    for (size_t x = 0; x < xsize; ++x) {
      const double muls[3] = { 30, 3.84, 30 };
      const size_t idx = y * xsize + x;
      const double a = (*suppression)[0][idx];
      const double b = (*suppression)[1][idx];
      const double c = (*suppression)[2][idx];
      (*suppression)[0][idx] = SuppressionRedMinusGreen(muls[0] * a + 0.1 * b);
      (*suppression)[1][idx] = SuppressionRedPlusGreen(muls[1] * b);
      (*suppression)[2][idx] = SuppressionBlue(muls[2] * c + 0.1 * b);
    }
  }
}

bool ButteraugliInterface(size_t xsize, size_t ysize,
                          const std::vector<std::vector<double> > &rgb0,
                          const std::vector<std::vector<double> > &rgb1,
                          std::vector<double> &diffmap,
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

}  // namespace butteraugli
