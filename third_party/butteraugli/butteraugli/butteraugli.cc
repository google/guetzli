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
//
// The physical architecture of butteraugli is based on the following naming
// convention:
//   * Opsin - dynamics of the photosensitive chemicals in the retina
//             with their immediate electrical processing
//   * Xyb - hybrid opponent/trichromatic color space
//     x is roughly red-subtract-green.
//     y is yellow.
//     b is blue.
//     Xyb values are computed from Opsin mixing, not directly from rgb.
//   * Mask - for visual masking
//   * Hf - color modeling for spatially high-frequency features
//   * Lf - color modeling for spatially low-frequency features
//   * Diffmap - to cluster and build an image of error between the images
//   * Blur - to hold the smoothing code

#include "butteraugli/butteraugli.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <array>


// Restricted pointers speed up Convolution(); MSVC uses a different keyword.
#ifdef _MSC_VER
#define __restrict__ __restrict
#endif

#ifndef PROFILER_ENABLED
#define PROFILER_ENABLED 0
#endif
#if PROFILER_ENABLED
#else
#define PROFILER_FUNC
#define PROFILER_ZONE(name)
#endif

namespace butteraugli {

void *CacheAligned::Allocate(const size_t bytes) {
  char *const allocated = static_cast<char *>(malloc(bytes + kCacheLineSize));
  if (allocated == nullptr) {
    return nullptr;
  }
  const uintptr_t misalignment =
      reinterpret_cast<uintptr_t>(allocated) & (kCacheLineSize - 1);
  // malloc is at least kPointerSize aligned, so we can store the "allocated"
  // pointer immediately before the aligned memory.
  assert(misalignment % kPointerSize == 0);
  char *const aligned = allocated + kCacheLineSize - misalignment;
  memcpy(aligned - kPointerSize, &allocated, kPointerSize);
  return BUTTERAUGLI_ASSUME_ALIGNED(aligned, 64);
}

void CacheAligned::Free(void *aligned_pointer) {
  if (aligned_pointer == nullptr) {
    return;
  }
  char *const aligned = static_cast<char *>(aligned_pointer);
  assert(reinterpret_cast<uintptr_t>(aligned) % kCacheLineSize == 0);
  char *allocated;
  memcpy(&allocated, aligned - kPointerSize, kPointerSize);
  assert(allocated <= aligned - kPointerSize);
  assert(allocated >= aligned - kCacheLineSize);
  free(allocated);
}

static inline bool IsNan(const float x) {
  uint32_t bits;
  memcpy(&bits, &x, sizeof(bits));
  const uint32_t bitmask_exp = 0x7F800000;
  return (bits & bitmask_exp) == bitmask_exp && (bits & 0x7FFFFF);
}

static inline bool IsNan(const double x) {
  uint64_t bits;
  memcpy(&bits, &x, sizeof(bits));
  return (0x7ff0000000000001ULL <= bits && bits <= 0x7fffffffffffffffULL) ||
         (0xfff0000000000001ULL <= bits && bits <= 0xffffffffffffffffULL);
}

static inline void CheckImage(const ImageF &image, const char *name) {
  for (size_t y = 0; y < image.ysize(); ++y) {
    const float * const BUTTERAUGLI_RESTRICT row = image.Row(y);
    for (size_t x = 0; x < image.xsize(); ++x) {
      if (IsNan(row[x])) {
        printf("Image %s @ %lu,%lu (of %lu,%lu)\n", name, x, y, image.xsize(),
               image.ysize());
        exit(1);
      }
    }
  }
}

#if BUTTERAUGLI_ENABLE_CHECKS

#define CHECK_NAN(x, str)                \
  do {                                   \
    if (IsNan(x)) {                      \
      printf("%d: %s\n", __LINE__, str); \
      abort();                           \
    }                                    \
  } while (0)

#define CHECK_IMAGE(image, name) CheckImage(image, name)

#else

#define CHECK_NAN(x, str)
#define CHECK_IMAGE(image, name)

#endif


// Purpose of kInternalGoodQualityThreshold:
// Normalize 'ok' image degradation to 1.0 across different versions of
// butteraugli.
static const double kInternalGoodQualityThreshold = 11.698467292807441;
static const double kGlobalScale = 1.0 / kInternalGoodQualityThreshold;

inline float DotProduct(const float u[3], const float v[3]) {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

std::vector<float> ComputeKernel(float sigma) {
  const float m = 2.25;  // Accuracy increases when m is increased.
  const float scaler = -1.0 / (2 * sigma * sigma);
  const int diff = std::max<int>(1, m * fabs(sigma));
  std::vector<float> kernel(2 * diff + 1);
  for (int i = -diff; i <= diff; ++i) {
    kernel[i + diff] = exp(scaler * i * i);
  }
  return kernel;
}

void ConvolveBorderColumn(
    const ImageF& in,
    const std::vector<float>& kernel,
    const float weight_no_border,
    const float border_ratio,
    const size_t x,
    float* const BUTTERAUGLI_RESTRICT row_out) {
  const int offset = kernel.size() / 2;
  int minx = x < offset ? 0 : x - offset;
  int maxx = std::min<int>(in.xsize() - 1, x + offset);
  float weight = 0.0f;
  for (int j = minx; j <= maxx; ++j) {
    weight += kernel[j - x + offset];
  }
  // Interpolate linearly between the no-border scaling and border scaling.
  weight = (1.0f - border_ratio) * weight + border_ratio * weight_no_border;
  float scale = 1.0f / weight;
  for (size_t y = 0; y < in.ysize(); ++y) {
    const float* const BUTTERAUGLI_RESTRICT row_in = in.Row(y);
    float sum = 0.0f;
    for (int j = minx; j <= maxx; ++j) {
      sum += row_in[j] * kernel[j - x + offset];
    }
    row_out[y] = sum * scale;
  }
}

// Computes a horizontal convolution and transposes the result.
ImageF Convolution(const ImageF& in,
                   const std::vector<float>& kernel,
                   const float border_ratio) {
  ImageF out(in.ysize(), in.xsize());
  const int len = kernel.size();
  const int offset = kernel.size() / 2;
  float weight_no_border = 0.0f;
  for (int j = 0; j < len; ++j) {
    weight_no_border += kernel[j];
  }
  float scale_no_border = 1.0f / weight_no_border;
  const int border1 = in.xsize() <= offset ? in.xsize() : offset;
  const int border2 = in.xsize() - offset;
  int x = 0;
  // left border
  for (; x < border1; ++x) {
    ConvolveBorderColumn(in, kernel, weight_no_border, border_ratio, x,
                         out.Row(x));
  }
  // middle
  for (; x < border2; ++x) {
    float* const BUTTERAUGLI_RESTRICT row_out = out.Row(x);
    for (size_t y = 0; y < in.ysize(); ++y) {
      const float* const BUTTERAUGLI_RESTRICT row_in = &in.Row(y)[x - offset];
      float sum = 0.0f;
      for (int j = 0; j < len; ++j) {
        sum += row_in[j] * kernel[j];
      }
      row_out[y] = sum * scale_no_border;
    }
  }
  // right border
  for (; x < in.xsize(); ++x) {
    ConvolveBorderColumn(in, kernel, weight_no_border, border_ratio, x,
                         out.Row(x));
  }
  return out;
}

// A blur somewhat similar to a 2D Gaussian blur.
// See: https://en.wikipedia.org/wiki/Gaussian_blur
ImageF Blur(const ImageF& in, float sigma, float border_ratio) {
  std::vector<float> kernel = ComputeKernel(sigma);
  return Convolution(Convolution(in, kernel, border_ratio),
                     kernel, border_ratio);
}

// DoGBlur is an approximate of difference of Gaussians. We use it to
// approximate LoG (Laplacian of Gaussians).
// See: https://en.wikipedia.org/wiki/Difference_of_Gaussians
// For motivation see:
// https://en.wikipedia.org/wiki/Pyramid_(image_processing)#Laplacian_pyramid
ImageF DoGBlur(const ImageF& in, float sigma, float border_ratio) {
  ImageF blur1 = Blur(in, sigma, border_ratio);
  ImageF blur2 = Blur(in, sigma * 2.0f, border_ratio);
  static const float mix = 0.5;
  ImageF out(in.xsize(), in.ysize());
  for (size_t y = 0; y < in.ysize(); ++y) {
    const float* const BUTTERAUGLI_RESTRICT row1 = blur1.Row(y);
    const float* const BUTTERAUGLI_RESTRICT row2 = blur2.Row(y);
    float* const BUTTERAUGLI_RESTRICT row_out = out.Row(y);
    for (size_t x = 0; x < in.xsize(); ++x) {
      row_out[x] = (1.0f + mix) * row1[x] - mix * row2[x];
    }
  }
  return out;
}

// Clamping linear interpolator.
inline double InterpolateClampNegative(const double *array,
                                       int size, double ix) {
  if (ix < 0) {
    ix = 0;
  }
  int baseix = static_cast<int>(ix);
  double res;
  if (baseix >= size - 1) {
    res = array[size - 1];
  } else {
    double mix = ix - baseix;
    int nextix = baseix + 1;
    res = array[baseix] + mix * (array[nextix] - array[baseix]);
  }
  return res;
}

double GammaMinArg() {
  double out0, out1, out2;
  OpsinAbsorbance(0.0, 0.0, 0.0, &out0, &out1, &out2);
  return std::min(out0, std::min(out1, out2));
}

double GammaMaxArg() {
  double out0, out1, out2;
  OpsinAbsorbance(255.0, 255.0, 255.0, &out0, &out1, &out2);
  return std::max(out0, std::max(out1, out2));
}

// The input images c0 and c1 include the high frequency component only.
// The output scalar images b0 and b1 include the correlation of Y and
// B component at a Gaussian locality around the respective pixel.
ImageF BlurredBlueCorrelation(const std::vector<ImageF>& uhf,
                              const std::vector<ImageF>& hf) {
  const size_t xsize = uhf[0].xsize();
  const size_t ysize = uhf[0].ysize();
  ImageF yb(xsize, ysize);
  ImageF yy(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    const float* const BUTTERAUGLI_RESTRICT row_uhf_y = uhf[1].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_uhf_b = uhf[2].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_hf_y = hf[1].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_hf_b = hf[2].Row(y);
    float* const BUTTERAUGLI_RESTRICT row_yb = yb.Row(y);
    float* const BUTTERAUGLI_RESTRICT row_yy = yy.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      const float yval = row_hf_y[x] + row_uhf_y[x];
      const float bval = row_hf_b[x] + row_uhf_b[x];
      row_yb[x] = yval * bval;
      row_yy[x] = yval * yval;
    }
  }
  const double kSigma = 8.48596332566;
  ImageF yy_blurred = Blur(yy, kSigma, 0.0);
  ImageF yb_blurred = Blur(yb, kSigma, 0.0);
  for (size_t y = 0; y < ysize; ++y) {
    const float* const BUTTERAUGLI_RESTRICT row_uhf_y = uhf[1].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_hf_y = hf[1].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_yy = yy_blurred.Row(y);
    float* const BUTTERAUGLI_RESTRICT row_yb = yb_blurred.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      static const float epsilon = 20.0101389159;
      const float yval = row_hf_y[x] + row_uhf_y[x];
      row_yb[x] *= yval / (row_yy[x] + epsilon);
    }
  }
  return yb_blurred;
}

double SimpleGamma(double v) {
  static const double kGamma = 0.372322653176;
  static const double limit = 37.8000499603;
  double bright = v - limit;
  if (bright >= 0) {
    static const double mul = 0.0950819040934;
    v -= bright * mul;
  }
  {
    static const double limit2 = 74.6154406429;
    double bright2 = v - limit2;
    if (bright2 >= 0) {
      static const double mul = 0.01;
      v -= bright2 * mul;
    }
  }
  {
    static const double limit2 = 82.8505938033;
    double bright2 = v - limit2;
    if (bright2 >= 0) {
      static const double mul = 0.0316722592629;
      v -= bright2 * mul;
    }
  }
  {
    static const double limit2 = 92.8505938033;
    double bright2 = v - limit2;
    if (bright2 >= 0) {
      static const double mul = 0.221249885752;
      v -= bright2 * mul;
    }
  }
  {
    static const double limit2 = 102.8505938033;
    double bright2 = v - limit2;
    if (bright2 >= 0) {
      static const double mul = 0.0402547853939;
      v -= bright2 * mul;
    }
  }
  {
    static const double limit2 = 112.8505938033;
    double bright2 = v - limit2;
    if (bright2 >= 0) {
      static const double mul = 0.021471798711500003;
      v -= bright2 * mul;
    }
  }
  static const double offset = 0.106544447664;
  static const double scale = 10.7950943969;
  double retval = scale * (offset + pow(v, kGamma));
  return retval;
}

static inline double Gamma(double v) {
  //return SimpleGamma(v);
  return GammaPolynomial(v);
}

std::vector<ImageF> OpsinDynamicsImage(const std::vector<ImageF>& rgb) {
  PROFILER_FUNC;
  std::vector<ImageF> xyb(3);
  std::vector<ImageF> blurred(3);
  const double kSigma = 1.44316781537;
  for (int i = 0; i < 3; ++i) {
    xyb[i] = ImageF(rgb[i].xsize(), rgb[i].ysize());
    blurred[i] = Blur(rgb[i], kSigma, 0.0f);
  }
  for (size_t y = 0; y < rgb[0].ysize(); ++y) {
    const float* const BUTTERAUGLI_RESTRICT row_r = rgb[0].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_g = rgb[1].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_b = rgb[2].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_blurred_r = blurred[0].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_blurred_g = blurred[1].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_blurred_b = blurred[2].Row(y);
    float* const BUTTERAUGLI_RESTRICT row_out_x = xyb[0].Row(y);
    float* const BUTTERAUGLI_RESTRICT row_out_y = xyb[1].Row(y);
    float* const BUTTERAUGLI_RESTRICT row_out_b = xyb[2].Row(y);
    for (size_t x = 0; x < rgb[0].xsize(); ++x) {
      float sensitivity[3];
      {
        // Calculate sensitivity based on the smoothed image gamma derivative.
        float pre_mixed0, pre_mixed1, pre_mixed2;
        OpsinAbsorbance(row_blurred_r[x], row_blurred_g[x], row_blurred_b[x],
                        &pre_mixed0, &pre_mixed1, &pre_mixed2);
        // TODO: use new polynomial to compute Gamma(x)/x derivative.
        sensitivity[0] = Gamma(pre_mixed0) / pre_mixed0;
        sensitivity[1] = Gamma(pre_mixed1) / pre_mixed1;
        sensitivity[2] = Gamma(pre_mixed2) / pre_mixed2;
      }
      float cur_mixed0, cur_mixed1, cur_mixed2;
      OpsinAbsorbance(row_r[x], row_g[x], row_b[x],
                      &cur_mixed0, &cur_mixed1, &cur_mixed2);
      cur_mixed0 *= sensitivity[0];
      cur_mixed1 *= sensitivity[1];
      cur_mixed2 *= sensitivity[2];
      RgbToXyb(cur_mixed0, cur_mixed1, cur_mixed2,
               &row_out_x[x], &row_out_y[x], &row_out_b[x]);
    }
  }
  return xyb;
}

// Make area around zero less important (remove it).
static BUTTERAUGLI_INLINE float RemoveRangeAroundZero(float w, float x) {
  return x > w ? x - w : x < -w ? x + w : 0.0f;
}

// Make area around zero more important (2x it until the limit).
static BUTTERAUGLI_INLINE float AmplifyRangeAroundZero(float w, float x) {
  return x > w ? x + w : x < -w ? x - w : 2.0f * x;
}

std::vector<ImageF> ModifyRangeAroundZero(const double warray[2],
                                          const std::vector<ImageF>& in) {
  std::vector<ImageF> out;
  for (int k = 0; k < 3; ++k) {
    ImageF plane(in[k].xsize(), in[k].ysize());
    for (int y = 0; y < plane.ysize(); ++y) {
      auto row_in = in[k].Row(y);
      auto row_out = plane.Row(y);
      if (k == 2) {
        memcpy(row_out, row_in, plane.xsize() * sizeof(row_out[0]));
      } else if (warray[k] >= 0) {
        const double w = warray[k];
        for (int x = 0; x < plane.xsize(); ++x) {
          row_out[x] = RemoveRangeAroundZero(w, row_in[x]);
        }
      } else {
        const double w = -warray[k];
        for (int x = 0; x < plane.xsize(); ++x) {
          row_out[x] = AmplifyRangeAroundZero(w, row_in[x]);
        }
      }
    }
    out.emplace_back(std::move(plane));
  }
  return out;
}

// XybLowFreqToVals converts from low-frequency XYB space to the 'vals' space.
// Vals space can be converted to L2-norm space (Euclidean and normalized)
// through visual masking.
template <class V>
BUTTERAUGLI_INLINE void XybLowFreqToVals(const V &x, const V &y, const V &b_arg,
                                         V *BUTTERAUGLI_RESTRICT valx,
                                         V *BUTTERAUGLI_RESTRICT valy,
                                         V *BUTTERAUGLI_RESTRICT valb) {
  static const double xmuli = 5.55938080599;
  static const double ymuli = 4.58944186612;
  static const double bmuli = 11.2394147993;
  static const double y_to_b_muli = -0.634050875917;

  const V xmul(xmuli);
  const V ymul(ymuli);
  const V bmul(bmuli);
  const V y_to_b_mul(y_to_b_muli);
  const V b = b_arg + y_to_b_mul * y;
  *valb = b * bmul;
  *valx = x * xmul;
  *valy = y * ymul;
}

static ImageF SuppressHfInBrightAreas(size_t xsize, size_t ysize,
                                      const ImageF& hf,
                                      const ImageF& brightness) {
  ImageF inew(xsize, ysize);
  static const float mul = 1.12879309857;
  static const float mul2 = 2.27308648104;
  static const float reg = 2000 * mul2;
  for (size_t y = 0; y < ysize; ++y) {
    const float* const rowhf = hf.Row(y);
    const float* const rowbr = brightness.Row(y);
    float* const rownew = inew.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      float v = rowhf[x];
      float scaler = mul * reg / (reg + rowbr[x]);
      rownew[x] = scaler * v;
    }
  }
  return inew;
}


static ImageF MaximumClamping(size_t xsize, size_t ysize, const ImageF& ix,
                              double yw) {
  ImageF inew(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    const float* const rowx = ix.Row(y);
    float* const rownew = inew.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      double v = rowx[x];
      if (v >= yw) {
        v -= yw;
        v *= 0.7;
        v += yw;
      } else if (v < -yw) {
        v += yw;
        v *= 0.7;
        v -= yw;
      }
      rownew[x] = v;
    }
  }
  return inew;
}

double Suppress(double x, double y) {
  static const double yw = 16.1797443814;
  static const double s = 0.512720106089;
  const double scaler = s + (yw * (1.0 - s)) / (yw + y * y);
  return scaler * x;
}

static ImageF SuppressXByY(size_t xsize, size_t ysize,
                           const ImageF& ix, const ImageF& iy, const double w) {
  ImageF inew(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    const float* const rowx = ix.Row(y);
    const float* const rowy = iy.Row(y);
    float* const rownew = inew.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      rownew[x] = Suppress(rowx[x], w * rowy[x]);
    }
  }
  return inew;
}

static void SeparateFrequencies(
    size_t xsize, size_t ysize,
    const std::vector<ImageF>& xyb,
    PsychoImage &ps) {
  PROFILER_FUNC;
  ps.lf.resize(3);
  ps.mf.resize(3);
  ps.hf.resize(3);
  ps.uhf.resize(3);
  for (int i = 0; i < 3; ++i) {
    // Extract lf ...
    static const double kSigmaLf = 7.41525493374;
    ps.lf[i] = DoGBlur(xyb[i], kSigmaLf, 0.0f);
    // ... and keep everything else in mf.
    ps.mf[i] = ImageF(xsize, ysize);
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        ps.mf[i].Row(y)[x] = xyb[i].Row(y)[x] - ps.lf[i].Row(y)[x];
      }
    }
    // Divide mf into mf and hf.
    static const double kSigmaHf = 0.5 * kSigmaLf;
    ps.hf[i] = ImageF(xsize, ysize);
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        ps.hf[i].Row(y)[x] = ps.mf[i].Row(y)[x];
      }
    }
    ps.mf[i] = DoGBlur(ps.mf[i], kSigmaHf, 0.0f);
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        ps.hf[i].Row(y)[x] -= ps.mf[i].Row(y)[x];
      }
    }
    // Divide hf into hf and uhf.
    static const double kSigmaUhf = 0.5 * kSigmaHf;
    ps.uhf[i] = ImageF(xsize, ysize);
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        ps.uhf[i].Row(y)[x] = ps.hf[i].Row(y)[x];
      }
    }
    ps.hf[i] = DoGBlur(ps.hf[i], kSigmaUhf, 0.0f);
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        ps.uhf[i].Row(y)[x] -= ps.hf[i].Row(y)[x];
      }
    }
  }
  // Modify range around zero code only concerns the high frequency
  // planes and only the X and Y channels.
  static const double uhf_xy_modification[2] = {
    -0.0262070567973,
    -5.07470663801,
  };
  static const double hf_xy_modification[2] = {
    0.0260892336622,
    -0.00789413170469,
  };
  static const double mf_xy_modification[2] = {
    0.0185433382632,
    -0.158111863182,
  };
  ps.uhf = ModifyRangeAroundZero(uhf_xy_modification, ps.uhf);
  ps.hf = ModifyRangeAroundZero(hf_xy_modification, ps.hf);
  ps.mf = ModifyRangeAroundZero(mf_xy_modification, ps.mf);
  // Convert low freq xyb to vals space so that we can do a simple squared sum
  // diff on the low frequencies later.
  for (size_t y = 0; y < ysize; ++y) {
    float* BUTTERAUGLI_RESTRICT const row_x = ps.lf[0].Row(y);
    float* BUTTERAUGLI_RESTRICT const row_y = ps.lf[1].Row(y);
    float* BUTTERAUGLI_RESTRICT const row_b = ps.lf[2].Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      float valx, valy, valb;
      XybLowFreqToVals(row_x[x], row_y[x], row_b[x], &valx, &valy, &valb);
      row_x[x] = valx;
      row_y[x] = valy;
      row_b[x] = valb;
    }
  }
  // Suppress red-green by intensity change.
  static const double suppress[3] = {
    -0.0636106621652,
    26.8144000514,
  };
  ps.uhf[0] = SuppressXByY(xsize, ysize, ps.uhf[0], ps.uhf[1], suppress[0]);
  ps.hf[0] = SuppressXByY(xsize, ysize, ps.hf[0], ps.hf[1], suppress[1]);
  static const double maxclamp0 = 0.670004157878;
  ps.uhf[0] = MaximumClamping(xsize, ysize, ps.uhf[0], maxclamp0);
  static const double maxclamp1 = 2.645076392;
  ps.hf[0] = MaximumClamping(xsize, ysize, ps.hf[0], maxclamp1);
  static const double maxclamp2 = 64.9667578444;
  ps.uhf[1] = MaximumClamping(xsize, ysize, ps.uhf[1], maxclamp2);
  static const double maxclamp3 = 79.5957602666;
  ps.hf[1] = MaximumClamping(xsize, ysize, ps.hf[1], maxclamp3);

  ps.hf[1] = SuppressHfInBrightAreas(xsize, ysize, ps.hf[1], ps.lf[1]);
  ps.uhf[1] = SuppressHfInBrightAreas(xsize, ysize, ps.uhf[1], ps.lf[1]);
  ps.mf[1] = SuppressHfInBrightAreas(xsize, ysize, ps.mf[1], ps.lf[1]);
}

static void SameNoiseLevelsX(const ImageF& i0, const ImageF& i1,
                             const double kSigma,
                             const double w,
                             const double maxclamp,
                             ImageF* BUTTERAUGLI_RESTRICT diffmap) {
  ImageF blurred0 = CopyPixels(i0);
  ImageF blurred1 = CopyPixels(i1);
  for (size_t y = 0; y < i0.ysize(); ++y) {
    float* BUTTERAUGLI_RESTRICT const row0 = blurred0.Row(y);
    float* BUTTERAUGLI_RESTRICT const row1 = blurred1.Row(y);
    for (size_t x = i0.xsize() - 1; x != 0; --x) {
      row0[x] -= row0[x - 1];
      row1[x] -= row1[x - 1];
      row0[x] = fabs(row0[x]);
      row1[x] = fabs(row1[x]);
      if (row0[x] > maxclamp) row0[x] = maxclamp;
      if (row1[x] > maxclamp) row1[x] = maxclamp;
    }
    row0[0] = 0.25 * row0[1];
    row1[0] = 0.25 * row0[1];
  }
  blurred0 = Blur(blurred0, kSigma, 0.0);
  blurred1 = Blur(blurred1, kSigma, 0.0);
  for (size_t y = 0; y < i0.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT const row0 = blurred0.Row(y);
    const float* BUTTERAUGLI_RESTRICT const row1 = blurred1.Row(y);
    float* BUTTERAUGLI_RESTRICT const row_diff = diffmap->Row(y);
    for (size_t x = 0; x < i0.xsize(); ++x) {
      double diff = row0[x] - row1[x];
      row_diff[x] += w * diff * diff;
    }
  }
}

static void SameNoiseLevelsY(const ImageF& i0, const ImageF& i1,
                             const double kSigma,
                             const double w,
                             const double maxclamp,
                             ImageF* BUTTERAUGLI_RESTRICT diffmap) {
  ImageF blurred0 = CopyPixels(i0);
  ImageF blurred1 = CopyPixels(i1);
  for (size_t y = i0.ysize() - 1; y != 0; --y) {
    float* BUTTERAUGLI_RESTRICT const row0prev = blurred0.Row(y - 1);
    float* BUTTERAUGLI_RESTRICT const row1prev = blurred1.Row(y - 1);
    float* BUTTERAUGLI_RESTRICT const row0 = blurred0.Row(y);
    float* BUTTERAUGLI_RESTRICT const row1 = blurred1.Row(y);
    for (size_t x = 0; x < i0.xsize(); ++x) {
      row0[x] -= row0prev[x];
      row1[x] -= row1prev[x];
      row0[x] = fabs(row0[x]);
      row1[x] = fabs(row1[x]);
      if (row0[x] > maxclamp) row0[x] = maxclamp;
      if (row1[x] > maxclamp) row1[x] = maxclamp;
    }
  }
  {
    float* BUTTERAUGLI_RESTRICT const row0 = blurred0.Row(0);
    float* BUTTERAUGLI_RESTRICT const row1 = blurred1.Row(0);
    float* BUTTERAUGLI_RESTRICT const row0next = blurred0.Row(1);
    float* BUTTERAUGLI_RESTRICT const row1next = blurred1.Row(1);
    for (size_t x = 0; x < i0.xsize(); ++x) {
      row0[x] = 0.25 * row0next[x];
      row1[x] = 0.25 * row1next[x];
    }
  }
  blurred0 = Blur(blurred0, kSigma, 0.0);
  blurred1 = Blur(blurred1, kSigma, 0.0);
  for (size_t y = 0; y < i0.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT const row0 = blurred0.Row(y);
    const float* BUTTERAUGLI_RESTRICT const row1 = blurred1.Row(y);
    float* BUTTERAUGLI_RESTRICT const row_diff = diffmap->Row(y);
    for (size_t x = 0; x < i0.xsize(); ++x) {
      double diff = row0[x] - row1[x];
      row_diff[x] += w * diff * diff;
    }
  }
}

static void SameNoiseLevelsYP1(const ImageF& i0, const ImageF& i1,
                               const double kSigma,
                               const double w,
                               const double maxclamp,
                               ImageF* BUTTERAUGLI_RESTRICT diffmap) {
  ImageF blurred0 = CopyPixels(i0);
  ImageF blurred1 = CopyPixels(i1);
  for (size_t y = i0.ysize() - 1; y != 0; --y) {
    float* BUTTERAUGLI_RESTRICT const row0prev = blurred0.Row(y - 1);
    float* BUTTERAUGLI_RESTRICT const row1prev = blurred1.Row(y - 1);
    float* BUTTERAUGLI_RESTRICT const row0 = blurred0.Row(y);
    float* BUTTERAUGLI_RESTRICT const row1 = blurred1.Row(y);
    for (size_t x = 1; x < i0.xsize(); ++x) {
      row0[x] -= row0prev[x - 1];
      row1[x] -= row1prev[x - 1];
      row0[x] = fabs(row0[x]);
      row1[x] = fabs(row1[x]);
      if (row0[x] > maxclamp) row0[x] = maxclamp;
      if (row1[x] > maxclamp) row1[x] = maxclamp;
    }
    row0[0] = 0.25 * row0[1];
    row1[0] = 0.25 * row1[1];
  }
  {
    float* BUTTERAUGLI_RESTRICT const row0 = blurred0.Row(0);
    float* BUTTERAUGLI_RESTRICT const row1 = blurred1.Row(0);
    float* BUTTERAUGLI_RESTRICT const row0next = blurred0.Row(1);
    float* BUTTERAUGLI_RESTRICT const row1next = blurred1.Row(1);
    for (size_t x = 0; x < i0.xsize(); ++x) {
      row0[x] = 0.25 * row0next[x];
      row1[x] = 0.25 * row1next[x];
    }
  }
  blurred0 = Blur(blurred0, kSigma, 0.0);
  blurred1 = Blur(blurred1, kSigma, 0.0);
  for (size_t y = 0; y < i0.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT const row0 = blurred0.Row(y);
    const float* BUTTERAUGLI_RESTRICT const row1 = blurred1.Row(y);
    float* BUTTERAUGLI_RESTRICT const row_diff = diffmap->Row(y);
    for (size_t x = 0; x < i0.xsize(); ++x) {
      double diff = row0[x] - row1[x];
      row_diff[x] += w * diff * diff;
    }
  }
}

static void SameNoiseLevelsYM1(const ImageF& i0, const ImageF& i1,
                               const double kSigma,
                               const double w,
                               const double maxclamp,
                               ImageF* BUTTERAUGLI_RESTRICT diffmap) {
  ImageF blurred0 = CopyPixels(i0);
  ImageF blurred1 = CopyPixels(i1);
  for (size_t y = i0.ysize() - 1; y != 0; --y) {
    float* BUTTERAUGLI_RESTRICT const row0prev = blurred0.Row(y - 1);
    float* BUTTERAUGLI_RESTRICT const row1prev = blurred1.Row(y - 1);
    float* BUTTERAUGLI_RESTRICT const row0 = blurred0.Row(y);
    float* BUTTERAUGLI_RESTRICT const row1 = blurred1.Row(y);
    for (size_t x = 0; x + 1 < i0.xsize(); ++x) {
      row0[x] -= row0prev[x + 1];
      row1[x] -= row1prev[x + 1];
      row0[x] = fabs(row0[x]);
      row1[x] = fabs(row1[x]);
      if (row0[x] > maxclamp) row0[x] = maxclamp;
      if (row1[x] > maxclamp) row1[x] = maxclamp;
    }
    row0[i0.xsize() - 1] = 0.25 * row0[i0.xsize() - 2];
    row1[i0.xsize() - 1] = 0.25 * row1[i0.xsize() - 2];
  }
  {
    float* BUTTERAUGLI_RESTRICT const row0 = blurred0.Row(0);
    float* BUTTERAUGLI_RESTRICT const row1 = blurred1.Row(0);
    float* BUTTERAUGLI_RESTRICT const row0next = blurred0.Row(1);
    float* BUTTERAUGLI_RESTRICT const row1next = blurred1.Row(1);
    for (size_t x = 0; x < i0.xsize(); ++x) {
      row0[x] = 0.25 * row0next[x];
      row1[x] = 0.25 * row1next[x];
    }
  }
  blurred0 = Blur(blurred0, kSigma, 0.0);
  blurred1 = Blur(blurred1, kSigma, 0.0);
  for (size_t y = 0; y < i0.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT const row0 = blurred0.Row(y);
    const float* BUTTERAUGLI_RESTRICT const row1 = blurred1.Row(y);
    float* BUTTERAUGLI_RESTRICT const row_diff = diffmap->Row(y);
    for (size_t x = 0; x < i0.xsize(); ++x) {
      double diff = row0[x] - row1[x];
      row_diff[x] += w * diff * diff;
    }
  }
}


static void L2Diff(const ImageF& i0, const ImageF& i1, const double w,
                   ImageF* BUTTERAUGLI_RESTRICT diffmap) {
  for (size_t y = 0; y < i0.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT const row0 = i0.Row(y);
    const float* BUTTERAUGLI_RESTRICT const row1 = i1.Row(y);
    float* BUTTERAUGLI_RESTRICT const row_diff = diffmap->Row(y);
    for (size_t x = 0; x < i0.xsize(); ++x) {
      double diff = row0[x] - row1[x];
      row_diff[x] += w * diff * diff;
    }
  }
}

static void LNDiff(const ImageF& i0, const ImageF& i1, const double w,
                   double n,
                   ImageF* BUTTERAUGLI_RESTRICT diffmap) {
  if (n == 1.0) {
    for (size_t y = 0; y < i0.ysize(); ++y) {
      const float* BUTTERAUGLI_RESTRICT const row0 = i0.Row(y);
      const float* BUTTERAUGLI_RESTRICT const row1 = i1.Row(y);
      float* BUTTERAUGLI_RESTRICT const row_diff = diffmap->Row(y);
      for (size_t x = 0; x < i0.xsize(); ++x) {
        double diff = fabs(row0[x] - row1[x]);
        row_diff[x] += w * diff;
      }
    }
  } else if (n == 2.0) {
    for (size_t y = 0; y < i0.ysize(); ++y) {
      const float* BUTTERAUGLI_RESTRICT const row0 = i0.Row(y);
      const float* BUTTERAUGLI_RESTRICT const row1 = i1.Row(y);
      float* BUTTERAUGLI_RESTRICT const row_diff = diffmap->Row(y);
      for (size_t x = 0; x < i0.xsize(); ++x) {
        double diff = row0[x] - row1[x];
        row_diff[x] += w * diff * diff;
      }
    }
  } else {
    for (size_t y = 0; y < i0.ysize(); ++y) {
      const float* BUTTERAUGLI_RESTRICT const row0 = i0.Row(y);
      const float* BUTTERAUGLI_RESTRICT const row1 = i1.Row(y);
      float* BUTTERAUGLI_RESTRICT const row_diff = diffmap->Row(y);
      for (size_t x = 0; x < i0.xsize(); ++x) {
        double diff = fabs(row0[x] - row1[x]);
        row_diff[x] += w * pow(diff, n);
      }
    }
  }
}

// Making a cluster of local errors to be more impactful than
// just a single error.
ImageF CalculateDiffmap(const ImageF& diffmap_in) {
  PROFILER_FUNC;
  // Take square root.
  ImageF diffmap(diffmap_in.xsize(), diffmap_in.ysize());
  static const float kInitialSlope = 100.0f;
  for (size_t y = 0; y < diffmap.ysize(); ++y) {
    const float* const BUTTERAUGLI_RESTRICT row_in = diffmap_in.Row(y);
    float* const BUTTERAUGLI_RESTRICT row_out = diffmap.Row(y);
    for (size_t x = 0; x < diffmap.xsize(); ++x) {
      const float orig_val = row_in[x];
      // TODO(b/29974893): Until that is fixed do not call sqrt on very small
      // numbers.
      row_out[x] = (orig_val < (1.0f / (kInitialSlope * kInitialSlope))
                    ? kInitialSlope * orig_val
                    : std::sqrt(orig_val));
    }
  }
  {
    static const double kSigma = 1.72547472444;
    static const double mul1 = 0.458794906198;
    static const float scale = 1.0f / (1.0f + mul1);
    static const double border_ratio = 1.0; // 2.01209066992;
    ImageF blurred = Blur(diffmap, kSigma, border_ratio);
    for (int y = 0; y < diffmap.ysize(); ++y) {
      const float* const BUTTERAUGLI_RESTRICT row_blurred = blurred.Row(y);
      float* const BUTTERAUGLI_RESTRICT row = diffmap.Row(y);
      for (int x = 0; x < diffmap.xsize(); ++x) {
        row[x] += mul1 * row_blurred[x];
        row[x] *= scale;
      }
    }
  }
  return diffmap;
}

void MaskPsychoImage(const PsychoImage& pi0, const PsychoImage& pi1,
                     const size_t xsize, const size_t ysize,
                     std::vector<ImageF>* BUTTERAUGLI_RESTRICT mask,
                     std::vector<ImageF>* BUTTERAUGLI_RESTRICT mask_dc) {
  std::vector<ImageF> mask_xyb0 = CreatePlanes<float>(xsize, ysize, 3);
  std::vector<ImageF> mask_xyb1 = CreatePlanes<float>(xsize, ysize, 3);
  static const double muls[4] = {
    0,
    1.75262681671,
    0.962073813832,
    2.587167299,
  };
  for (int i = 0; i < 2; ++i) {
    double a = muls[2 * i];
    double b = muls[2 * i + 1];
    for (size_t y = 0; y < ysize; ++y) {
      const float* const BUTTERAUGLI_RESTRICT row_hf0 = pi0.hf[i].Row(y);
      const float* const BUTTERAUGLI_RESTRICT row_hf1 = pi1.hf[i].Row(y);
      const float* const BUTTERAUGLI_RESTRICT row_uhf0 = pi0.uhf[i].Row(y);
      const float* const BUTTERAUGLI_RESTRICT row_uhf1 = pi1.uhf[i].Row(y);
      float* const BUTTERAUGLI_RESTRICT row0 = mask_xyb0[i].Row(y);
      float* const BUTTERAUGLI_RESTRICT row1 = mask_xyb1[i].Row(y);
      for (size_t x = 0; x < xsize; ++x) {
        row0[x] = a * row_uhf0[x] + b * row_hf0[x];
        row1[x] = a * row_uhf1[x] + b * row_hf1[x];
      }
    }
  }
  Mask(mask_xyb0, mask_xyb1, mask, mask_dc);
}

ButteraugliComparator::ButteraugliComparator(const std::vector<ImageF>& rgb0)
    : xsize_(rgb0[0].xsize()),
      ysize_(rgb0[0].ysize()),
      num_pixels_(xsize_ * ysize_) {
  if (xsize_ < 8 || ysize_ < 8) return;
  std::vector<ImageF> xyb0 = OpsinDynamicsImage(rgb0);
  SeparateFrequencies(xsize_, ysize_, xyb0, pi0_);
}

void ButteraugliComparator::Mask(
    std::vector<ImageF>* BUTTERAUGLI_RESTRICT mask,
    std::vector<ImageF>* BUTTERAUGLI_RESTRICT mask_dc) const {
  MaskPsychoImage(pi0_, pi0_, xsize_, ysize_, mask, mask_dc);
}

void ButteraugliComparator::Diffmap(const std::vector<ImageF>& rgb1,
                                    ImageF &result) const {
  PROFILER_FUNC;
  if (xsize_ < 8 || ysize_ < 8) return;
  DiffmapOpsinDynamicsImage(OpsinDynamicsImage(rgb1), result);
}

void ButteraugliComparator::DiffmapOpsinDynamicsImage(
    const std::vector<ImageF>& xyb1,
    ImageF &result) const {
  PROFILER_FUNC;
  if (xsize_ < 8 || ysize_ < 8) return;
  PsychoImage pi1;
  SeparateFrequencies(xsize_, ysize_, xyb1, pi1);
  result = ImageF(xsize_, ysize_);
  DiffmapPsychoImage(pi1, result);
}

void ButteraugliComparator::DiffmapPsychoImage(const PsychoImage& pi1,
                                               ImageF& result) const {
  PROFILER_FUNC;
  if (xsize_ < 8 || ysize_ < 8) {
    return;
  }
  std::vector<ImageF> block_diff_dc(3);
  std::vector<ImageF> block_diff_ac(3);
  for (int c = 0; c < 3; ++c) {
    block_diff_dc[c] = ImageF(xsize_, ysize_, 0.0);
    block_diff_ac[c] = ImageF(xsize_, ysize_, 0.0);
  }

  static const double wUhfMalta = 1.23657307981;
  static const double norm1Uhf = 466.149933668;
  MaltaDiffMap(pi0_.uhf[1], pi1.uhf[1], wUhfMalta, norm1Uhf,
               &block_diff_ac[1]);

  static const double wUhfMaltaX = 3.36199686627;
  static const double norm1UhfX = norm1Uhf;
  MaltaDiffMap(pi0_.uhf[0], pi1.uhf[0], wUhfMaltaX, norm1UhfX,
               &block_diff_ac[0]);

  static const double wHfMalta = 15.6469934822;
  static const double norm1Hf = norm1Uhf;
  MaltaDiffMap(pi0_.hf[1], pi1.hf[1], wHfMalta, norm1Hf,
               &block_diff_ac[1]);

  static const double wHfMaltaX = 129.122071602;
  static const double norm1HfX = norm1Uhf;
  MaltaDiffMap(pi0_.hf[0], pi1.hf[0], wHfMaltaX, norm1HfX,
               &block_diff_ac[0]);

  static const double wMfMaltaX = 51.2720081112;
  static const double norm1MfX = norm1Uhf;
  MaltaDiffMap(pi0_.mf[0], pi1.mf[0], wMfMaltaX, norm1MfX,
               &block_diff_ac[0]);

  static const double wmul[11] = {
    0,
    2.52211854569,
    0,
    0,
    7.34229797917,
    1.92307717196,
    0.779146234988,
    4.91012468367,
    1.83755854086,
    0.0,
    234.519844745,
  };


  static const double maxclamp = 72.6815019479;
  static const double kSigmaHfX = 10.8163829574;
  SameNoiseLevelsX(pi0_.hf[1], pi1.hf[1], kSigmaHfX, wmul[10], maxclamp,
                   &block_diff_ac[1]);
  SameNoiseLevelsY(pi0_.hf[1], pi1.hf[1], kSigmaHfX, wmul[10], maxclamp,
                   &block_diff_ac[1]);
  SameNoiseLevelsYP1(pi0_.hf[1], pi1.hf[1], kSigmaHfX, wmul[10], maxclamp,
                     &block_diff_ac[1]);
  SameNoiseLevelsYM1(pi0_.hf[1], pi1.hf[1], kSigmaHfX, wmul[10], maxclamp,
                     &block_diff_ac[1]);


  static const double valn[9] = {
    2.0,
    2.0,
    2.0,
    2.0,
    1.0,
    2.0,
    2.0,
    2.0,
    2.0,
  };

  for (int c = 0; c < 3; ++c) {
    if (wmul[c] != 0) {
      LNDiff(pi0_.hf[c], pi1.hf[c], wmul[c], valn[c], &block_diff_ac[c]);
    }
    LNDiff(pi0_.mf[c], pi1.mf[c], wmul[3 + c], valn[3 + c], &block_diff_ac[c]);
    LNDiff(pi0_.lf[c], pi1.lf[c], wmul[6 + c], valn[6 + c], &block_diff_dc[c]);
  }

  static const double wBlueCorr = 0.0122171286852;
  ImageF blurred_b_y_correlation0 = BlurredBlueCorrelation(pi0_.uhf, pi0_.hf);
  ImageF blurred_b_y_correlation1 = BlurredBlueCorrelation(pi1.uhf, pi1.hf);
  L2Diff(blurred_b_y_correlation0, blurred_b_y_correlation1, wBlueCorr,
         &block_diff_ac[2]);

  std::vector<ImageF> mask_xyb;
  std::vector<ImageF> mask_xyb_dc;
  MaskPsychoImage(pi0_, pi1, xsize_, ysize_, &mask_xyb, &mask_xyb_dc);

  result = CalculateDiffmap(
      CombineChannels(mask_xyb, mask_xyb_dc, block_diff_dc, block_diff_ac));
}

static float MaltaUnit(const float *d, const int xs) {
  const int xs3 = 3 * xs;
  float retval = 0;
  static const float kEdgemul = 0.0309255573587;
  {
    // x grows, y constant
    float sum =
        d[-4] +
        d[-3] +
        d[-2] +
        d[-1] +
        d[0] +
        d[1] +
        d[2] +
        d[3] +
        d[4];
    retval += sum * sum;
    float sum2 =
        d[xs - 4] +
        d[xs - 3] +
        d[xs - 2] +
        d[xs - 1] +
        d[xs] +
        d[xs + 1] +
        d[xs + 2] +
        d[xs + 3] +
        d[xs + 4];
    float edge = sum - sum2;
    retval += kEdgemul * edge * edge;
  }
  {
    // y grows, x constant
    float sum =
        d[-xs3 - xs] +
        d[-xs3] +
        d[-xs - xs] +
        d[-xs] +
        d[0] +
        d[xs] +
        d[xs + xs] +
        d[xs3] +
        d[xs3 + xs];
    retval += sum * sum;
    float sum2 =
        d[-xs3 - xs + 1] +
        d[-xs3 + 1] +
        d[-xs - xs + 1] +
        d[-xs + 1] +
        d[1] +
        d[xs + 1] +
        d[xs + xs + 1] +
        d[xs3 + 1] +
        d[xs3 + xs + 1];
    float edge = sum - sum2;
    retval += kEdgemul * edge * edge;
  }
  {
    // both grow
    float sum =
        d[-xs3 - 3] +
        d[-xs - xs - 2] +
        d[-xs - 1] +
        d[0] +
        d[xs + 1] +
        d[xs + xs + 2] +
        d[xs3 + 3];
    retval += sum * sum;
  }
  {
    // y grows, x shrinks
    float sum =
        d[-xs3 + 3] +
        d[-xs - xs + 2] +
        d[-xs + 1] +
        d[0] +
        d[xs - 1] +
        d[xs + xs - 2] +
        d[xs3 - 3];
    retval += sum * sum;
  }
  {
    // y grows -4 to 4, x shrinks 1 -> -1
    float sum =
        d[-xs3 - xs + 1] +
        d[-xs3 + 1] +
        d[-xs - xs + 1] +
        d[-xs] +
        d[0] +
        d[xs] +
        d[xs - 1] +
        d[xs3 - 1] +
        d[xs3 + xs - 1];
    retval += sum * sum;
  }
  {
    //  y grows -4 to 4, x grows -1 -> 1
    float sum =
        d[-xs3 - xs - 1] +
        d[-xs3 - 1] +
        d[-xs - xs - 1] +
        d[-xs] +
        d[0] +
        d[xs] +
        d[xs + 1] +
        d[xs3 + 1] +
        d[xs3 + xs + 1];
    retval += sum * sum;
  }
  {
    // x grows -4 to 4, y grows -1 to 1
    float sum =
        d[-4 - xs] +
        d[-3 - xs] +
        d[-2 - xs] +
        d[-1] +
        d[0] +
        d[1] +
        d[2 + xs] +
        d[3 + xs] +
        d[4 + xs];
    retval += sum * sum;
  }
  {
    // x grows -4 to 4, y shrinks 1 to -1
    float sum =
        d[-4 + xs] +
        d[-3 + xs] +
        d[-2 + xs] +
        d[-1] +
        d[0] +
        d[1] +
        d[2 - xs] +
        d[3 - xs] +
        d[4 - xs];
    retval += sum * sum;
  }
  {
    /* 0_________
       1__*______
       2___*_____
       3___*_____
       4____0____
       5_____*___
       6_____*___
       7______*__
       8_________ */
    float sum =
        d[-xs3 - 2] +
        d[-xs - xs - 1] +
        d[-xs - 1] +
        d[0] +
        d[xs + 1] +
        d[xs + xs + 1] +
        d[xs3 + 2];
    retval += sum * sum;
  }
  {
    /* 0_________
       1______*__
       2_____*___
       3_____*___
       4____0____
       5___*_____
       6___*_____
       7__*______
       8_________ */
    float sum =
        d[-xs3 + 2] +
        d[-xs - xs + 1] +
        d[-xs + 1] +
        d[0] +
        d[xs - 1] +
        d[xs + xs - 1] +
        d[xs3 - 2];
    retval += sum * sum;
  }
  {
    /* 0_________
       1_________
       2_*_______
       3__**_____
       4____0____
       5_____**__
       6_______*_
       7_________
       8_________ */
    float sum =
        d[-xs - xs - 3] +
        d[-xs - 2] +
        d[-xs - 1] +
        d[0] +
        d[xs + 1] +
        d[xs + 2] +
        d[xs + xs + 3];
    retval += sum * sum;
  }
  {
    /* 0_________
       1_________
       2_______*_
       3_____**__
       4____0____
       5__**_____
       6_*_______
       7_________
       8_________ */
    float sum =
        d[-xs - xs + 3] +
        d[-xs + 2] +
        d[-xs + 1] +
        d[0] +
        d[xs - 1] +
        d[xs - 2] +
        d[xs + xs - 3];
    retval += sum * sum;
  }
  {
    /* 0_________
       1_________
       2_________
       3______**_
       4____0*___
       5__**_____
       6**_______
       7_________
       8_________ */

    float sum =
        d[xs + xs - 4] +
        d[xs + xs - 3] +
        d[xs - 2] +
        d[xs - 1] +
        d[0] +
        d[1] +
        d[-xs + 2] +
        d[-xs + 3];
    retval += sum * sum;
  }
  {
    /* 0_________
       1_________
       2**_______
       3__**_____
       4____0*___
       5______**_
       6_________
       7_________
       8_________ */
    float sum =
        d[-xs - xs - 4] +
        d[-xs - xs - 3] +
        d[-xs - 2] +
        d[-xs - 1] +
        d[0] +
        d[1] +
        d[xs + 2] +
        d[xs + 3];
    retval += sum * sum;
  }
  {
    /* 0__*______
       1__*______
       2___*_____
       3___*_____
       4____0____
       5____*____
       6_____*___
       7_____*___
       8_________ */
    float sum =
        d[-xs3 - xs - 2] +
        d[-xs3 - 2] +
        d[-xs - xs - 1] +
        d[-xs - 1] +
        d[0] +
        d[xs] +
        d[xs + xs + 1] +
        d[xs3 + 1];
    retval += sum * sum;
  }
  {
    /* 0______*__
       1______*__
       2_____*___
       3_____*___
       4____0____
       5____*____
       6___*_____
       7___*_____
       8_________ */
    float sum =
        d[-xs3 - xs + 2] +
        d[-xs3 + 2] +
        d[-xs - xs + 1] +
        d[-xs + 1] +
        d[0] +
        d[xs] +
        d[xs + xs - 1] +
        d[xs3 - 1];
    retval += sum * sum;
  }
  return retval;
}

void ButteraugliComparator::MaltaDiffMap(
    const ImageF& y0, const ImageF& y1,
    const double weight,
    const double norm1,
    ImageF* BUTTERAUGLI_RESTRICT block_diff_ac) const {
  PROFILER_FUNC;
  const double len = 3.75;
  static const double mulli = 0.414888221144;
  const double w = mulli * sqrt(weight) / (len * 2 + 1);
  const double norm2 = w * norm1;
  std::vector<float> diffs(ysize_ * xsize_);
  std::vector<float> sums(ysize_ * xsize_);
  for (size_t y = 0, ix = 0; y < ysize_; ++y) {
    const float* BUTTERAUGLI_RESTRICT const row0 = y0.Row(y);
    const float* BUTTERAUGLI_RESTRICT const row1 = y1.Row(y);
    for (size_t x = 0; x < xsize_; ++x, ++ix) {
      double absval = 0.5 * (std::abs(row0[x]) + std::abs(row1[x]));
      double diff = row0[x] - row1[x];
      double scaler = norm2 / (norm1 + absval);
      diffs[ix] = scaler * diff;
    }
  }
  float borderimage[9 * 9];
  for (size_t y0 = 0; y0 < ysize_; ++y0) {
    float* const BUTTERAUGLI_RESTRICT row_diff = block_diff_ac->Row(y0);
    const bool fastModeY = y0 >= 4 && y0 < ysize_ - 4;
    for (size_t x0 = 0; x0 < xsize_; ++x0) {
      int ix0 = y0 * xsize_ + x0;
      const float *d = &diffs[ix0];
      const bool fastModeX = x0 >= 4 && x0 < xsize_ - 4;
      if (fastModeY && fastModeX) {
        row_diff[x0] += MaltaUnit(d, xsize_);
      } else {
        for (int dy = 0; dy < 9; ++dy) {
          int y = y0 + dy - 4;
          if (y < 0 || y >= ysize_) {
            for (int dx = 0; dx < 9; ++dx) {
              borderimage[dy * 9 + dx] = 0;
            }
          } else {
            for (int dx = 0; dx < 9; ++dx) {
              int x = x0 + dx - 4;
              if (x < 0 || x >= xsize_) {
                borderimage[dy * 9 + dx] = 0;
              } else {
                borderimage[dy * 9 + dx] = diffs[y * xsize_ + x];
              }
            }
          }
        }
        row_diff[x0] += MaltaUnit(&borderimage[4 * 9 + 4], 9);
      }
    }
  }
}

ImageF ButteraugliComparator::CombineChannels(
    const std::vector<ImageF>& mask_xyb,
    const std::vector<ImageF>& mask_xyb_dc,
    const std::vector<ImageF>& block_diff_dc,
    const std::vector<ImageF>& block_diff_ac) const {
  PROFILER_FUNC;
  ImageF result(xsize_, ysize_);
  for (size_t y = 0; y < ysize_; ++y) {
    float* const BUTTERAUGLI_RESTRICT row_out = result.Row(y);
    for (size_t x = 0; x < xsize_; ++x) {
      float mask[3];
      float dc_mask[3];
      float diff_dc[3];
      float diff_ac[3];
      for (int i = 0; i < 3; ++i) {
        mask[i] = mask_xyb[i].Row(y)[x];
        dc_mask[i] = mask_xyb_dc[i].Row(y)[x];
        diff_dc[i] = block_diff_dc[i].Row(y)[x];
        diff_ac[i] = block_diff_ac[i].Row(y)[x];
      }
      row_out[x] = (DotProduct(diff_dc, dc_mask) + DotProduct(diff_ac, mask));
    }
  }
  return result;
}

double ButteraugliScoreFromDiffmap(const ImageF& diffmap) {
  PROFILER_FUNC;
  float retval = 0.0f;
  for (size_t y = 0; y < diffmap.ysize(); ++y) {
    const float * const BUTTERAUGLI_RESTRICT row = diffmap.Row(y);
    for (size_t x = 0; x < diffmap.xsize(); ++x) {
      retval = std::max(retval, row[x]);
    }
  }
  return retval;
}

#include <stdio.h>

// ===== Functions used by Mask only =====
static std::array<double, 512> MakeMask(
    double extmul, double extoff,
    double mul, double offset,
    double scaler) {
  std::array<double, 512> lut;
  for (int i = 0; i < lut.size(); ++i) {
    const double c = mul / ((0.01 * scaler * i) + offset);
    lut[i] = kGlobalScale * (1.0 + extmul * (c + extoff));
    if (lut[i] < 1e-5) {
      lut[i] = 1e-5;
    }
    assert(lut[i] >= 0.0);
    lut[i] *= lut[i];
  }
  return lut;
}

double MaskX(double delta) {
  PROFILER_FUNC;
  static const double extmul = 2.52662693217;
  static const double extoff = 2.0577595478;
  static const double offset = 0.342502406734;
  static const double scaler = 14.4867545374;
  static const double mul = 6.03009840821;
  static const std::array<double, 512> lut =
                MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

double MaskY(double delta) {
  PROFILER_FUNC;
  static const double extmul = 0.965276993931;
  static const double extoff = -0.613819681771;
  static const double offset = 1.40903146071;
  static const double scaler = 1.07806168416;
  static const double mul = 7.09705888614;
  static const std::array<double, 512> lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

double MaskDcX(double delta) {
  PROFILER_FUNC;
  static const double extmul = 10.8596436398;
  static const double extoff = 1.58374126704;
  static const double offset = 0.651968473749;
  static const double scaler = 519.45682322;
  static const double mul = 4.72871406401;
  static const std::array<double, 512> lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

double MaskDcY(double delta) {
  PROFILER_FUNC;
  static const double extmul = 0.00538280872633;
  static const double extoff = 59.04237604;
  static const double offset = 0.0474092064444;
  static const double scaler = 5.52679307489;
  static const double mul = 22.7326511523;
  static const std::array<double, 512> lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

ImageF DiffPrecompute(const ImageF& xyb0, const ImageF& xyb1) {
  PROFILER_FUNC;
  const size_t xsize = xyb0.xsize();
  const size_t ysize = xyb0.ysize();
  ImageF result(xsize, ysize);
  size_t x2, y2;
  for (size_t y = 0; y < ysize; ++y) {
    if (y + 1 < ysize) {
      y2 = y + 1;
    } else if (y > 0) {
      y2 = y - 1;
    } else {
      y2 = y;
    }
    const float* const BUTTERAUGLI_RESTRICT row0_in = xyb0.Row(y);
    const float* const BUTTERAUGLI_RESTRICT row1_in = xyb1.Row(y);
    const float* const BUTTERAUGLI_RESTRICT row0_in2 = xyb0.Row(y2);
    const float* const BUTTERAUGLI_RESTRICT row1_in2 = xyb1.Row(y2);
    float* const BUTTERAUGLI_RESTRICT row_out = result.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      if (x + 1 < xsize) {
        x2 = x + 1;
      } else if (x > 0) {
        x2 = x - 1;
      } else {
        x2 = x;
      }
      double sup0 = (fabs(row0_in[x] - row0_in[x2]) +
                     fabs(row0_in[x] - row0_in2[x]));
      double sup1 = (fabs(row1_in[x] - row1_in[x2]) +
                     fabs(row1_in[x] - row1_in2[x]));
      static const double mul0 = 0.972407512222;
      row_out[x] = mul0 * std::min(sup0, sup1);
      static const double cutoff = 123.915065832;
      if (row_out[x] >= cutoff) {
        row_out[x] = cutoff;
      }
    }
  }
  return result;
}

void Mask(const std::vector<ImageF>& xyb0,
          const std::vector<ImageF>& xyb1,
          std::vector<ImageF>* BUTTERAUGLI_RESTRICT mask,
          std::vector<ImageF>* BUTTERAUGLI_RESTRICT mask_dc) {
  PROFILER_FUNC;
  const size_t xsize = xyb0[0].xsize();
  const size_t ysize = xyb0[0].ysize();
  mask->resize(3);
  *mask_dc = CreatePlanes<float>(xsize, ysize, 3);
  double muls[4] = {
    0.05,
    0.144577484346,
    0.231880902493,
    0.529175844348,
  };
  double normalizer[2] = {
    1.0 / (muls[0] + muls[1]),
    1.0 / (muls[2] + muls[3]),
  };
  static const double r0 = 2.32030744494;
  static const double r1 = 7.55507439878;
  for (int i = 0; i < 2; ++i) {
    (*mask)[i] = ImageF(xsize, ysize);
    ImageF diff = DiffPrecompute(xyb0[i], xyb1[i]);
    ImageF blurred1 = Blur(diff, r0, 0.0f);
    ImageF blurred2 = Blur(diff, r1, 0.0f);
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        const double val = normalizer[i] * (
            muls[2 * i + 0] * blurred1.Row(y)[x] +
            muls[2 * i + 1] * blurred2.Row(y)[x]);
        (*mask)[i].Row(y)[x] = val;
      }
    }
  }
  (*mask)[2] = ImageF(xsize, ysize);
  static const double mul[2] = {
    12.5378252408,
    2.31907764902,
  };
  static const double w00 = 9.27537465315;
  static const double w11 = 2.64039747911;
  static const double w_ytob_hf = 1.06493691683;
  static const double w_ytob_lf = 9.71657276893;
  static const double p1_to_p0 = 0.0153146912176;

  for (size_t y = 0; y < ysize; ++y) {
    for (size_t x = 0; x < xsize; ++x) {
      const double s0 = (*mask)[0].Row(y)[x];
      const double s1 = (*mask)[1].Row(y)[x];
      const double p1 = mul[1] * w11 * s1;
      const double p0 = mul[0] * w00 * s0 + p1_to_p0 * p1;

      (*mask)[0].Row(y)[x] = MaskX(p0);
      (*mask)[1].Row(y)[x] = MaskY(p1);
      (*mask)[2].Row(y)[x] = w_ytob_hf * MaskY(p1);
      (*mask_dc)[0].Row(y)[x] = MaskDcX(p0);
      (*mask_dc)[1].Row(y)[x] = MaskDcY(p1);
      (*mask_dc)[2].Row(y)[x] = w_ytob_lf * MaskDcY(p1);
    }
  }
}

void ButteraugliDiffmap(const std::vector<ImageF> &rgb0_image,
                        const std::vector<ImageF> &rgb1_image,
                        ImageF &result_image) {
  const size_t xsize = rgb0_image[0].xsize();
  const size_t ysize = rgb0_image[0].ysize();
  static const int kMax = 8;
  if (xsize < kMax || ysize < kMax) {
    // Butteraugli values for small (where xsize or ysize is smaller
    // than 8 pixels) images are non-sensical, but most likely it is
    // less disruptive to try to compute something than just give up.
    // Temporarily extend the borders of the image to fit 8 x 8 size.
    int xborder = xsize < kMax ? (kMax - xsize) / 2 : 0;
    int yborder = ysize < kMax ? (kMax - ysize) / 2 : 0;
    size_t xscaled = std::max<size_t>(kMax, xsize);
    size_t yscaled = std::max<size_t>(kMax, ysize);
    std::vector<ImageF> scaled0 = CreatePlanes<float>(xscaled, yscaled, 3);
    std::vector<ImageF> scaled1 = CreatePlanes<float>(xscaled, yscaled, 3);
    for (int i = 0; i < 3; ++i) {
      for (int y = 0; y < yscaled; ++y) {
        for (int x = 0; x < xscaled; ++x) {
          size_t x2 = std::min<size_t>(xsize - 1, std::max(0, x - xborder));
          size_t y2 = std::min<size_t>(ysize - 1, std::max(0, y - yborder));
          scaled0[i].Row(y)[x] = rgb0_image[i].Row(y2)[x2];
          scaled1[i].Row(y)[x] = rgb1_image[i].Row(y2)[x2];
        }
      }
    }
    ImageF diffmap_scaled;
    ButteraugliDiffmap(scaled0, scaled1, diffmap_scaled);
    result_image = ImageF(xsize, ysize);
    for (int y = 0; y < ysize; ++y) {
      for (int x = 0; x < xsize; ++x) {
        result_image.Row(y)[x] = diffmap_scaled.Row(y + yborder)[x + xborder];
      }
    }
    return;
  }
  ButteraugliComparator butteraugli(rgb0_image);
  butteraugli.Diffmap(rgb1_image, result_image);
}

bool ButteraugliInterface(const std::vector<ImageF> &rgb0,
                          const std::vector<ImageF> &rgb1,
                          ImageF &diffmap,
                          double &diffvalue) {
  const size_t xsize = rgb0[0].xsize();
  const size_t ysize = rgb0[0].ysize();
  if (xsize < 1 || ysize < 1) {
    return false;  // No image.
  }
  for (int i = 1; i < 3; i++) {
    if (rgb0[i].xsize() != xsize || rgb0[i].ysize() != ysize ||
        rgb1[i].xsize() != xsize || rgb1[i].ysize() != ysize) {
      return false;  // Image planes must have same dimensions.
    }
  }
  ButteraugliDiffmap(rgb0, rgb1, diffmap);
  diffvalue = ButteraugliScoreFromDiffmap(diffmap);
  return true;
}

bool ButteraugliAdaptiveQuantization(size_t xsize, size_t ysize,
    const std::vector<std::vector<float> > &rgb, std::vector<float> &quant) {
  if (xsize < 16 || ysize < 16) {
    return false;  // Butteraugli is undefined for small images.
  }
  size_t size = xsize * ysize;

  std::vector<ImageF> rgb_planes = PlanesFromPacked(xsize, ysize, rgb);
  std::vector<ImageF> scale_xyb;
  std::vector<ImageF> scale_xyb_dc;
  Mask(rgb_planes, rgb_planes, &scale_xyb, &scale_xyb_dc);
  quant.reserve(size);

  // Mask gives us values in 3 color channels, but for now we take only
  // the intensity channel.
  for (size_t y = 0; y < ysize; ++y) {
    for (size_t x = 0; x < xsize; ++x) {
      quant.push_back(scale_xyb[1].Row(y)[x]);
    }
  }
  return true;
}

double ButteraugliFuzzyClass(double score) {
  static const double fuzzy_width_up = 6.78721575514;
  static const double fuzzy_width_down = 5.96507193294;
  static const double m0 = 2.0;
  static const double scaler = 0.861077627013;
  double val;
  if (score < 1.0) {
    // val in [scaler .. 2.0]
    val = m0 / (1.0 + exp((score - 1.0) * fuzzy_width_down));
    val -= 1.0;  // from [1 .. 2] to [0 .. 1]
    val *= 2.0 - scaler;  // from [0 .. 1] to [0 .. 2.0 - scaler]
    val += scaler;  // from [0 .. 2.0 - scaler] to [scaler .. 2.0]
  } else {
    // val in [0 .. scaler]
    val = m0 / (1.0 + exp((score - 1.0) * fuzzy_width_up));
    val *= scaler;
  }
  return val;
}

double ButteraugliFuzzyInverse(double seek) {
  double pos = 0;
  for (double range = 1.0; range >= 1e-10; range *= 0.5) {
    double cur = ButteraugliFuzzyClass(pos);
    if (cur < seek) {
      pos -= range;
    } else {
      pos += range;
    }
  }
  return pos;
}

namespace {

void ScoreToRgb(double score, double good_threshold, double bad_threshold,
                uint8_t rgb[3]) {
  double heatmap[12][3] = {
      {0, 0, 0},
      {0, 0, 1},
      {0, 1, 1},
      {0, 1, 0},  // Good level
      {1, 1, 0},
      {1, 0, 0},  // Bad level
      {1, 0, 1},
      {0.5, 0.5, 1.0},
      {1.0, 0.5, 0.5},  // Pastel colors for the very bad quality range.
      {1.0, 1.0, 0.5},
      {
          1, 1, 1,
      },
      {
          1, 1, 1,
      },
  };
  if (score < good_threshold) {
    score = (score / good_threshold) * 0.3;
  } else if (score < bad_threshold) {
    score = 0.3 +
            (score - good_threshold) / (bad_threshold - good_threshold) * 0.15;
  } else {
    score = 0.45 + (score - bad_threshold) / (bad_threshold * 12) * 0.5;
  }
  static const int kTableSize = sizeof(heatmap) / sizeof(heatmap[0]);
  score = std::min<double>(std::max<double>(score * (kTableSize - 1), 0.0),
                           kTableSize - 2);
  int ix = static_cast<int>(score);
  double mix = score - ix;
  for (int i = 0; i < 3; ++i) {
    double v = mix * heatmap[ix + 1][i] + (1 - mix) * heatmap[ix][i];
    rgb[i] = static_cast<uint8_t>(255 * pow(v, 0.5) + 0.5);
  }
}

}  // namespace

void CreateHeatMapImage(const std::vector<float>& distmap,
                        double good_threshold, double bad_threshold,
                        size_t xsize, size_t ysize,
                        std::vector<uint8_t>* heatmap) {
  heatmap->resize(3 * xsize * ysize);
  for (size_t y = 0; y < ysize; ++y) {
    for (size_t x = 0; x < xsize; ++x) {
      int px = xsize * y + x;
      double d = distmap[px];
      uint8_t* rgb = &(*heatmap)[3 * px];
      ScoreToRgb(d, good_threshold, bad_threshold, rgb);
    }
  }
}

}  // namespace butteraugli
