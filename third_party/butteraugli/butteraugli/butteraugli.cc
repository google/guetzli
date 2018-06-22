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
static const double kInternalGoodQualityThreshold = 20.35;
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
  std::vector<float> scaled_kernel = kernel;
  for (int i = 0; i < scaled_kernel.size(); ++i) {
    scaled_kernel[i] *= scale_no_border;
  }
  // left border
  for (int x = 0; x < border1; ++x) {
    ConvolveBorderColumn(in, kernel, weight_no_border, border_ratio, x,
                         out.Row(x));
  }
  // middle
  for (size_t y = 0; y < in.ysize(); ++y) {
    const float* const BUTTERAUGLI_RESTRICT row_in = in.Row(y);
    for (int x = border1; x < border2; ++x) {
      const int d = x - offset;
      float* const BUTTERAUGLI_RESTRICT row_out = out.Row(x);
      float sum = 0.0f;
      for (int j = 0; j < len; ++j) {
        sum += row_in[d + j] * scaled_kernel[j];
      }
      row_out[y] = sum;
    }
  }
  // right border
  for (int x = border2; x < in.xsize(); ++x) {
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
  const double kSigma = 1.2;
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

// XybLowFreqToVals converts from low-frequency XYB space to the 'vals' space.
// Vals space can be converted to L2-norm space (Euclidean and normalized)
// through visual masking.
template <class V>
BUTTERAUGLI_INLINE void XybLowFreqToVals(const V &x, const V &y, const V &b_arg,
                                         V *BUTTERAUGLI_RESTRICT valx,
                                         V *BUTTERAUGLI_RESTRICT valy,
                                         V *BUTTERAUGLI_RESTRICT valb) {
  static const double xmuli = 5.57547552483;
  static const double ymuli = 1.20828034498;
  static const double bmuli = 6.08319517575;
  static const double y_to_b_muli = -0.628811683685;

  const V xmul(xmuli);
  const V ymul(ymuli);
  const V bmul(bmuli);
  const V y_to_b_mul(y_to_b_muli);
  const V b = b_arg + y_to_b_mul * y;
  *valb = b * bmul;
  *valx = x * xmul;
  *valy = y * ymul;
}

static ImageF SuppressInBrightAreas(size_t xsize, size_t ysize,
                                    double mul, double mul2, double reg,
                                    const ImageF& hf,
                                    const ImageF& brightness) {
  ImageF inew(xsize, ysize);
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


static float SuppressHfInBrightAreas(float hf, float brightness,
                                     float mul, float reg) {
  float scaler = mul * reg / (reg + brightness);
  return scaler * hf;
}

static float SuppressUhfInBrightAreas(float hf, float brightness,
                                      float mul, float reg) {
  float scaler = mul * reg / (reg + brightness);
  return scaler * hf;
}

static float MaximumClamp(float v, float maxval) {
  static const double kMul = 0.688059627878;
  if (v >= maxval) {
    v -= maxval;
    v *= kMul;
    v += maxval;
  } else if (v < -maxval) {
    v += maxval;
    v *= kMul;
    v -= maxval;
  }
  return v;
}

static ImageF MaximumClamping(size_t xsize, size_t ysize, const ImageF& ix,
                              double yw) {
  static const double kMul = 0.688059627878;
  ImageF inew(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    const float* const rowx = ix.Row(y);
    float* const rownew = inew.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      double v = rowx[x];
      if (v >= yw) {
        v -= yw;
        v *= kMul;
        v += yw;
      } else if (v < -yw) {
        v += yw;
        v *= kMul;
        v -= yw;
      }
      rownew[x] = v;
    }
  }
  return inew;
}

static ImageF SuppressXByY(size_t xsize, size_t ysize,
                           const ImageF& ix, const ImageF& iy,
                           const double yw) {
  static const double s = 0.745954517135;
  ImageF inew(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    const float* const rowx = ix.Row(y);
    const float* const rowy = iy.Row(y);
    float* const rownew = inew.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      const double xval = rowx[x];
      const double yval = rowy[x];
      const double scaler = s + (yw * (1.0 - s)) / (yw + yval * yval);
      rownew[x] = scaler * xval;
    }
  }
  return inew;
}

static void SeparateFrequencies(
    size_t xsize, size_t ysize,
    const std::vector<ImageF>& xyb,
    PsychoImage &ps) {
  PROFILER_FUNC;
  ps.lf.resize(3);   // XYB
  ps.mf.resize(3);   // XYB
  ps.hf.resize(2);   // XY
  ps.uhf.resize(2);  // XY
  // Extract lf ...
  static const double kSigmaLf = 7.46953768697;
  static const double kSigmaHf = 3.734768843485;
  static const double kSigmaUhf = 1.8673844217425;
  // At borders we move some more of the energy to the high frequency
  // parts, because there can be unfortunate continuations in tiling
  // background color etc. So we want to represent the borders with
  // some more accuracy.
  static double border_lf = -0.00457628248637;
  static double border_mf = -0.271277366628;
  static double border_hf = 0.147068973249;
  for (int i = 0; i < 3; ++i) {
    ps.lf[i] = Blur(xyb[i], kSigmaLf, border_lf);
    // ... and keep everything else in mf.
    ps.mf[i] = ImageF(xsize, ysize);
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        ps.mf[i].Row(y)[x] = xyb[i].Row(y)[x] - ps.lf[i].Row(y)[x];
      }
    }
    if (i == 2) {
      ps.mf[i] = Blur(ps.mf[i], kSigmaHf, border_mf);
      break;
    }
    // Divide mf into mf and hf.
    ps.hf[i] = ImageF(xsize, ysize);
    for (size_t y = 0; y < ysize; ++y) {
      float* BUTTERAUGLI_RESTRICT const row_mf = ps.mf[i].Row(y);
      float* BUTTERAUGLI_RESTRICT const row_hf = ps.hf[i].Row(y);
      for (size_t x = 0; x < xsize; ++x) {
        row_hf[x] = row_mf[x];
      }
    }
    ps.mf[i] = Blur(ps.mf[i], kSigmaHf, border_mf);
    static const double w0 = 0.120079806822;
    static const double w1 = 0.03430529365;
    if (i == 0) {
      for (size_t y = 0; y < ysize; ++y) {
        float* BUTTERAUGLI_RESTRICT const row_mf = ps.mf[0].Row(y);
        float* BUTTERAUGLI_RESTRICT const row_hf = ps.hf[0].Row(y);
        for (size_t x = 0; x < xsize; ++x) {
          row_hf[x] -= row_mf[x];
          row_mf[x] = RemoveRangeAroundZero(w0, row_mf[x]);
        }
      }
    } else {
      for (size_t y = 0; y < ysize; ++y) {
        float* BUTTERAUGLI_RESTRICT const row_mf = ps.mf[1].Row(y);
        float* BUTTERAUGLI_RESTRICT const row_hf = ps.hf[1].Row(y);
        for (size_t x = 0; x < xsize; ++x) {
          row_hf[x] -= row_mf[x];
          row_mf[x] = AmplifyRangeAroundZero(w1, row_mf[x]);
        }
      }
    }
  }
  // Suppress red-green by intensity change in the high freq channels.
  static const double suppress = 2.96534974403;
  ps.hf[0] = SuppressXByY(xsize, ysize, ps.hf[0], ps.hf[1], suppress);

  for (int i = 0; i < 2; ++i) {
    // Divide hf into hf and uhf.
    ps.uhf[i] = ImageF(xsize, ysize);
    for (size_t y = 0; y < ysize; ++y) {
      float* BUTTERAUGLI_RESTRICT const row_uhf = ps.uhf[i].Row(y);
      float* BUTTERAUGLI_RESTRICT const row_hf = ps.hf[i].Row(y);
      for (size_t x = 0; x < xsize; ++x) {
        row_uhf[x] = row_hf[x];
      }
    }
    ps.hf[i] = Blur(ps.hf[i], kSigmaUhf, border_hf);
    static const double kRemoveHfRange = 0.0287615200377;
    static const double kMaxclampHf = 78.8223237675;
    static const double kMaxclampUhf = 5.8907152736;
    static const float kMulSuppressHf = 1.10684769012;
    static const float kMulRegHf = 0.478741530298;
    static const float kRegHf = 2000 * kMulRegHf;
    static const float kMulSuppressUhf = 1.76905001176;
    static const float kMulRegUhf = 0.310148420674;
    static const float kRegUhf = 2000 * kMulRegUhf;

    if (i == 0) {
      for (size_t y = 0; y < ysize; ++y) {
        float* BUTTERAUGLI_RESTRICT const row_uhf = ps.uhf[0].Row(y);
        float* BUTTERAUGLI_RESTRICT const row_hf = ps.hf[0].Row(y);
        for (size_t x = 0; x < xsize; ++x) {
          row_uhf[x] -= row_hf[x];
          row_hf[x] = RemoveRangeAroundZero(kRemoveHfRange, row_hf[x]);
        }
      }
    } else {
      for (size_t y = 0; y < ysize; ++y) {
        float* BUTTERAUGLI_RESTRICT const row_uhf = ps.uhf[1].Row(y);
        float* BUTTERAUGLI_RESTRICT const row_hf = ps.hf[1].Row(y);
        float* BUTTERAUGLI_RESTRICT const row_lf = ps.lf[1].Row(y);
        for (size_t x = 0; x < xsize; ++x) {
          row_uhf[x] -= row_hf[x];
          row_hf[x] = MaximumClamp(row_hf[x], kMaxclampHf);
          row_uhf[x] = MaximumClamp(row_uhf[x], kMaxclampUhf);
          row_uhf[x] = SuppressUhfInBrightAreas(row_uhf[x], row_lf[x],
                                                kMulSuppressUhf, kRegUhf);
          row_hf[x] = SuppressHfInBrightAreas(row_hf[x], row_lf[x],
                                              kMulSuppressHf, kRegHf);

        }
      }
    }
  }
  // Modify range around zero code only concerns the high frequency
  // planes and only the X and Y channels.
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
}

static void SameNoiseLevels(const ImageF& i0, const ImageF& i1,
                            const double kSigma,
                            const double w,
                            const double maxclamp,
                            ImageF* BUTTERAUGLI_RESTRICT diffmap) {
  ImageF blurred(i0.xsize(), i0.ysize());
  for (size_t y = 0; y < i0.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT const row0 = i0.Row(y);
    const float* BUTTERAUGLI_RESTRICT const row1 = i1.Row(y);
    float* BUTTERAUGLI_RESTRICT const to = blurred.Row(y);
    for (size_t x = 0; x < i0.xsize(); ++x) {
      double v0 = fabs(row0[x]);
      double v1 = fabs(row1[x]);
      if (v0 > maxclamp) v0 = maxclamp;
      if (v1 > maxclamp) v1 = maxclamp;
      to[x] = v0 - v1;
    }

  }
  blurred = Blur(blurred, kSigma, 0.0);
  for (size_t y = 0; y < i0.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT const row = blurred.Row(y);
    float* BUTTERAUGLI_RESTRICT const row_diff = diffmap->Row(y);
    for (size_t x = 0; x < i0.xsize(); ++x) {
      double diff = row[x];
      row_diff[x] += w * diff * diff;
    }
  }
}

static void L2Diff(const ImageF& i0, const ImageF& i1, const double w,
                   ImageF* BUTTERAUGLI_RESTRICT diffmap) {
  if (w == 0) {
    return;
  }
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

// i0 is the original image.
// i1 is the deformed copy.
static void L2DiffAsymmetric(const ImageF& i0, const ImageF& i1,
                             double w_0gt1,
                             double w_0lt1,
                             ImageF* BUTTERAUGLI_RESTRICT diffmap) {
  if (w_0gt1 == 0 && w_0lt1 == 0) {
    return;
  }
  w_0gt1 *= 0.8;
  w_0lt1 *= 0.8;
  for (size_t y = 0; y < i0.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT const row0 = i0.Row(y);
    const float* BUTTERAUGLI_RESTRICT const row1 = i1.Row(y);
    float* BUTTERAUGLI_RESTRICT const row_diff = diffmap->Row(y);
    for (size_t x = 0; x < i0.xsize(); ++x) {
      // Primary symmetric quadratic objective.
      double diff = row0[x] - row1[x];
      row_diff[x] += w_0gt1 * diff * diff;

      // Secondary half-open quadratic objectives.
      const double fabs0 = fabs(row0[x]);
      const double too_small = 0.4 * fabs0;
      const double too_big = 1.0 * fabs0;

      if (row0[x] < 0) {
        if (row1[x] > -too_small) {
          double v = row1[x] + too_small;
          row_diff[x] += w_0lt1 * v * v;
        } else if (row1[x] < -too_big) {
          double v = -row1[x] - too_big;
          row_diff[x] += w_0lt1 * v * v;
        }
      } else {
        if (row1[x] < too_small) {
          double v = too_small - row1[x];
          row_diff[x] += w_0lt1 * v * v;
        } else if (row1[x] > too_big) {
          double v = row1[x] - too_big;
          row_diff[x] += w_0lt1 * v * v;
        }
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
    1.64178305129,
    0.831081703362,
    3.23680933546,
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
  const float hf_asymmetry_ = 0.8f;
  if (xsize_ < 8 || ysize_ < 8) {
    return;
  }
  std::vector<ImageF> block_diff_dc(3);
  std::vector<ImageF> block_diff_ac(3);
  for (int c = 0; c < 3; ++c) {
    block_diff_dc[c] = ImageF(xsize_, ysize_, 0.0);
    block_diff_ac[c] = ImageF(xsize_, ysize_, 0.0);
  }

  static const double wUhfMalta = 5.1409625726;
  static const double norm1Uhf = 58.5001247061;
  MaltaDiffMap(pi0_.uhf[1], pi1.uhf[1],
               wUhfMalta * hf_asymmetry_,
               wUhfMalta / hf_asymmetry_,
               norm1Uhf,
               &block_diff_ac[1]);

  static const double wUhfMaltaX = 4.91743441556;
  static const double norm1UhfX = 687196.39002;
  MaltaDiffMap(pi0_.uhf[0], pi1.uhf[0],
               wUhfMaltaX * hf_asymmetry_,
               wUhfMaltaX / hf_asymmetry_,
               norm1UhfX,
               &block_diff_ac[0]);

  static const double wHfMalta = 153.671655716;
  static const double norm1Hf = 83150785.9592;
  MaltaDiffMapLF(pi0_.hf[1], pi1.hf[1],
                 wHfMalta * sqrt(hf_asymmetry_),
                 wHfMalta / sqrt(hf_asymmetry_),
                 norm1Hf,
                 &block_diff_ac[1]);

  static const double wHfMaltaX = 668.358918152;
  static const double norm1HfX = 0.882954368025;
  MaltaDiffMapLF(pi0_.hf[0], pi1.hf[0],
                 wHfMaltaX * sqrt(hf_asymmetry_),
                 wHfMaltaX / sqrt(hf_asymmetry_),
                 norm1HfX,
                 &block_diff_ac[0]);

  static const double wMfMalta = 6841.81248144;
  static const double norm1Mf = 0.0135134962487;
  MaltaDiffMapLF(pi0_.mf[1], pi1.mf[1], wMfMalta, wMfMalta, norm1Mf,
                 &block_diff_ac[1]);

  static const double wMfMaltaX = 813.901703816;
  static const double norm1MfX = 16792.9322251;
  MaltaDiffMapLF(pi0_.mf[0], pi1.mf[0], wMfMaltaX, wMfMaltaX, norm1MfX,
                 &block_diff_ac[0]);

  static const double wmul[9] = {
    0,
    32.4449876135,
    0,
    0,
    0,
    0,
    1.01370836411,
    0,
    1.74566011615,
  };

  static const double maxclamp = 85.7047444518;
  static const double kSigmaHfX = 10.6666499623;
  static const double w = 884.809801415;
  SameNoiseLevels(pi0_.hf[1], pi1.hf[1], kSigmaHfX, w, maxclamp,
                  &block_diff_ac[1]);

  for (int c = 0; c < 3; ++c) {
    if (c < 2) {
      L2DiffAsymmetric(pi0_.hf[c], pi1.hf[c],
                       wmul[c] * hf_asymmetry_,
                       wmul[c] / hf_asymmetry_,
                       &block_diff_ac[c]);
    }
    L2Diff(pi0_.mf[c], pi1.mf[c], wmul[3 + c], &block_diff_ac[c]);
    L2Diff(pi0_.lf[c], pi1.lf[c], wmul[6 + c], &block_diff_dc[c]);
  }

  std::vector<ImageF> mask_xyb;
  std::vector<ImageF> mask_xyb_dc;
  MaskPsychoImage(pi0_, pi1, xsize_, ysize_, &mask_xyb, &mask_xyb_dc);

  result = CalculateDiffmap(
      CombineChannels(mask_xyb, mask_xyb_dc, block_diff_dc, block_diff_ac));
}

// Allows PaddedMaltaUnit to call either function via overloading.
struct MaltaTagLF {};
struct MaltaTag {};

static float MaltaUnit(MaltaTagLF, const float* BUTTERAUGLI_RESTRICT d,
                       const int xs) {
  const int xs3 = 3 * xs;
  float retval = 0;
  {
    // x grows, y constant
    float sum =
        d[-4] +
        d[-2] +
        d[0] +
        d[2] +
        d[4];
    retval += sum * sum;
  }
  {
    // y grows, x constant
    float sum =
        d[-xs3 - xs] +
        d[-xs - xs] +
        d[0] +
        d[xs + xs] +
        d[xs3 + xs];
    retval += sum * sum;
  }
  {
    // both grow
    float sum =
        d[-xs3 - 3] +
        d[-xs - xs - 2] +
        d[0] +
        d[xs + xs + 2] +
        d[xs3 + 3];
    retval += sum * sum;
  }
  {
    // y grows, x shrinks
    float sum =
        d[-xs3 + 3] +
        d[-xs - xs + 2] +
        d[0] +
        d[xs + xs - 2] +
        d[xs3 - 3];
    retval += sum * sum;
  }
  {
    // y grows -4 to 4, x shrinks 1 -> -1
    float sum =
        d[-xs3 - xs + 1] +
        d[-xs - xs + 1] +
        d[0] +
        d[xs + xs - 1] +
        d[xs3 + xs - 1];
    retval += sum * sum;
  }
  {
    //  y grows -4 to 4, x grows -1 -> 1
    float sum =
        d[-xs3 - xs - 1] +
        d[-xs - xs - 1] +
        d[0] +
        d[xs + xs + 1] +
        d[xs3 + xs + 1];
    retval += sum * sum;
  }
  {
    // x grows -4 to 4, y grows -1 to 1
    float sum =
        d[-4 - xs] +
        d[-2 - xs] +
        d[0] +
        d[2 + xs] +
        d[4 + xs];
    retval += sum * sum;
  }
  {
    // x grows -4 to 4, y shrinks 1 to -1
    float sum =
        d[-4 + xs] +
        d[-2 + xs] +
        d[0] +
        d[2 - xs] +
        d[4 - xs];
    retval += sum * sum;
  }
  {
    /* 0_________
       1__*______
       2___*_____
       3_________
       4____0____
       5_________
       6_____*___
       7______*__
       8_________ */
    float sum =
        d[-xs3 - 2] +
        d[-xs - xs - 1] +
        d[0] +
        d[xs + xs + 1] +
        d[xs3 + 2];
    retval += sum * sum;
  }
  {
    /* 0_________
       1______*__
       2_____*___
       3_________
       4____0____
       5_________
       6___*_____
       7__*______
       8_________ */
    float sum =
        d[-xs3 + 2] +
        d[-xs - xs + 1] +
        d[0] +
        d[xs + xs - 1] +
        d[xs3 - 2];
    retval += sum * sum;
  }
  {
    /* 0_________
       1_________
       2_*_______
       3__*______
       4____0____
       5______*__
       6_______*_
       7_________
       8_________ */
    float sum =
        d[-xs - xs - 3] +
        d[-xs - 2] +
        d[0] +
        d[xs + 2] +
        d[xs + xs + 3];
    retval += sum * sum;
  }
  {
    /* 0_________
       1_________
       2_______*_
       3______*__
       4____0____
       5__*______
       6_*_______
       7_________
       8_________ */
    float sum =
        d[-xs - xs + 3] +
        d[-xs + 2] +
        d[0] +
        d[xs - 2] +
        d[xs + xs - 3];
    retval += sum * sum;
  }
  {
    /* 0_________
       1_________
       2________*
       3______*__
       4____0____
       5__*______
       6*________
       7_________
       8_________ */

    float sum =
        d[xs + xs - 4] +
        d[xs - 2] +
        d[0] +
        d[-xs + 2] +
        d[-xs - xs + 4];
    retval += sum * sum;
  }
  {
    /* 0_________
       1_________
       2*________
       3__*______
       4____0____
       5______*__
       6________*
       7_________
       8_________ */
    float sum =
        d[-xs - xs - 4] +
        d[-xs - 2] +
        d[0] +
        d[xs + 2] +
        d[xs + xs + 4];
    retval += sum * sum;
  }
  {
    /* 0__*______
       1_________
       2___*_____
       3_________
       4____0____
       5_________
       6_____*___
       7_________
       8______*__ */
    float sum =
        d[-xs3 - xs - 2] +
        d[-xs - xs - 1] +
        d[0] +
        d[xs + xs + 1] +
        d[xs3 + xs + 2];
    retval += sum * sum;
  }
  {
    /* 0______*__
       1_________
       2_____*___
       3_________
       4____0____
       5_________
       6___*_____
       7_________
       8__*______ */
    float sum =
        d[-xs3 - xs + 2] +
        d[-xs - xs + 1] +
        d[0] +
        d[xs + xs - 1] +
        d[xs3 + xs - 2];
    retval += sum * sum;
  }
  return retval;
}

static float MaltaUnit(MaltaTag, const float* BUTTERAUGLI_RESTRICT d,
                       const int xs) {
  const int xs3 = 3 * xs;
  float retval = 0;
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
        d[xs + xs - 1] +
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
        d[xs + xs + 1] +
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

// Returns MaltaUnit. "fastMode" avoids bounds-checks when x0 and y0 are known
// to be far enough from the image borders.
template <bool fastMode, class Tag>
static BUTTERAUGLI_INLINE float PaddedMaltaUnit(
    float* const BUTTERAUGLI_RESTRICT diffs, const size_t x0, const size_t y0,
    const size_t xsize_, const size_t ysize_) {
  int ix0 = y0 * xsize_ + x0;
  const float* BUTTERAUGLI_RESTRICT d = &diffs[ix0];
  if (fastMode ||
      (x0 >= 4 && y0 >= 4 && x0 < (xsize_ - 4) && y0 < (ysize_ - 4))) {
    return MaltaUnit(Tag(), d, xsize_);
  }

  float borderimage[9 * 9];
  for (int dy = 0; dy < 9; ++dy) {
    int y = y0 + dy - 4;
    if (y < 0 || y >= ysize_) {
      for (int dx = 0; dx < 9; ++dx) {
        borderimage[dy * 9 + dx] = 0.0f;
      }
    } else {
      for (int dx = 0; dx < 9; ++dx) {
        int x = x0 + dx - 4;
        if (x < 0 || x >= xsize_) {
          borderimage[dy * 9 + dx] = 0.0f;
        } else {
          borderimage[dy * 9 + dx] = diffs[y * xsize_ + x];
        }
      }
    }
  }
  return MaltaUnit(Tag(), &borderimage[4 * 9 + 4], 9);
}

template <class Tag>
static void MaltaDiffMapImpl(const ImageF& lum0, const ImageF& lum1,
                             const size_t xsize_, const size_t ysize_,
                             const double w_0gt1,
                             const double w_0lt1,
                             double norm1,
                             const double len, const double mulli,
                             ImageF* block_diff_ac) {
  const float kWeight0 = 0.5;
  const float kWeight1 = 0.33;

  const double w_pre0gt1 = mulli * sqrt(kWeight0 * w_0gt1) / (len * 2 + 1);
  const double w_pre0lt1 = mulli * sqrt(kWeight1 * w_0lt1) / (len * 2 + 1);
  const float norm2_0gt1 = w_pre0gt1 * norm1;
  const float norm2_0lt1 = w_pre0lt1 * norm1;

  std::vector<float> diffs(ysize_ * xsize_);
  for (size_t y = 0, ix = 0; y < ysize_; ++y) {
    const float* BUTTERAUGLI_RESTRICT const row0 = lum0.Row(y);
    const float* BUTTERAUGLI_RESTRICT const row1 = lum1.Row(y);
    for (size_t x = 0; x < xsize_; ++x, ++ix) {
      const float absval = 0.5 * std::abs(row0[x]) + 0.5 * std::abs(row1[x]);
      const float diff = row0[x] - row1[x];
      const float scaler = norm2_0gt1 / (static_cast<float>(norm1) + absval);

      // Primary symmetric quadratic objective.
      diffs[ix] = scaler * diff;

      const float scaler2 = norm2_0lt1 / (static_cast<float>(norm1) + absval);
      const double fabs0 = fabs(row0[x]);

      // Secondary half-open quadratic objectives.
      const double too_small = 0.55 * fabs0;
      const double too_big = 1.05 * fabs0;

      if (row0[x] < 0) {
        if (row1[x] > -too_small) {
          double impact = scaler2 * (row1[x] + too_small);
          if (diff < 0) {
            diffs[ix] -= impact;
          } else {
            diffs[ix] += impact;
          }
        } else if (row1[x] < -too_big) {
          double impact = scaler2 * (-row1[x] - too_big);
          if (diff < 0) {
            diffs[ix] -= impact;
          } else {
            diffs[ix] += impact;
          }
        }
      } else {
        if (row1[x] < too_small) {
          double impact = scaler2 * (too_small - row1[x]);
          if (diff < 0) {
            diffs[ix] -= impact;
          } else {
            diffs[ix] += impact;
          }
        } else if (row1[x] > too_big) {
          double impact = scaler2 * (row1[x] - too_big);
          if (diff < 0) {
            diffs[ix] -= impact;
          } else {
            diffs[ix] += impact;
          }
        }
      }
    }
  }

  size_t y0 = 0;
  // Top
  for (; y0 < 4; ++y0) {
    float* const BUTTERAUGLI_RESTRICT row_diff = block_diff_ac->Row(y0);
    for (size_t x0 = 0; x0 < xsize_; ++x0) {
      row_diff[x0] +=
          PaddedMaltaUnit<false, Tag>(&diffs[0], x0, y0, xsize_, ysize_);
    }
  }

  // Middle
  for (; y0 < ysize_ - 4; ++y0) {
    float* const BUTTERAUGLI_RESTRICT row_diff = block_diff_ac->Row(y0);
    size_t x0 = 0;
    for (; x0 < 4; ++x0) {
      row_diff[x0] +=
          PaddedMaltaUnit<false, Tag>(&diffs[0], x0, y0, xsize_, ysize_);
    }
    for (; x0 < xsize_ - 4; ++x0) {
      row_diff[x0] +=
          PaddedMaltaUnit<true, Tag>(&diffs[0], x0, y0, xsize_, ysize_);
    }

    for (; x0 < xsize_; ++x0) {
      row_diff[x0] +=
          PaddedMaltaUnit<false, Tag>(&diffs[0], x0, y0, xsize_, ysize_);
    }
  }

  // Bottom
  for (; y0 < ysize_; ++y0) {
    float* const BUTTERAUGLI_RESTRICT row_diff = block_diff_ac->Row(y0);
    for (size_t x0 = 0; x0 < xsize_; ++x0) {
      row_diff[x0] +=
          PaddedMaltaUnit<false, Tag>(&diffs[0], x0, y0, xsize_, ysize_);
    }
  }
}

void ButteraugliComparator::MaltaDiffMap(
    const ImageF& lum0, const ImageF& lum1,
    const double w_0gt1,
    const double w_0lt1,
    const double norm1, ImageF* BUTTERAUGLI_RESTRICT block_diff_ac) const {
  PROFILER_FUNC;
  const double len = 3.75;
  static const double mulli = 0.354191303559;
  MaltaDiffMapImpl<MaltaTag>(lum0, lum1, xsize_, ysize_, w_0gt1, w_0lt1,
                             norm1, len,
                             mulli, block_diff_ac);
}

void ButteraugliComparator::MaltaDiffMapLF(
    const ImageF& lum0, const ImageF& lum1,
    const double w_0gt1,
    const double w_0lt1,
    const double norm1, ImageF* BUTTERAUGLI_RESTRICT block_diff_ac) const {
  PROFILER_FUNC;
  const double len = 3.75;
  static const double mulli = 0.405371989604;
  MaltaDiffMapImpl<MaltaTagLF>(lum0, lum1, xsize_, ysize_,
                               w_0gt1, w_0lt1,
                               norm1, len,
                               mulli, block_diff_ac);
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
  static const double extmul = 2.59885507073;
  static const double extoff = 3.08805636789;
  static const double offset = 0.315424196682;
  static const double scaler = 16.2770141832;
  static const double mul = 5.62939030582;
  static const std::array<double, 512> lut =
                MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

double MaskY(double delta) {
  static const double extmul = 0.9613705131;
  static const double extoff = -0.581933100068;
  static const double offset = 1.00846207765;
  static const double scaler = 2.2342321176;
  static const double mul = 6.64307621174;
  static const std::array<double, 512> lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

double MaskDcX(double delta) {
  static const double extmul = 10.0470705878;
  static const double extoff = 3.18472654033;
  static const double offset = 0.0551512255218;
  static const double scaler = 70.0;
  static const double mul = 0.373092999662;
  static const std::array<double, 512> lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

double MaskDcY(double delta) {
  static const double extmul = 0.0115640939227;
  static const double extoff = 45.9483175519;
  static const double offset = 0.0142290066313;
  static const double scaler = 5.0;
  static const double mul = 2.52611324247;
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
      static const double mul0 = 0.918416534734;
      row_out[x] = mul0 * std::min(sup0, sup1);
      static const double cutoff = 55.0184555849;
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
  double muls[2] = {
    0.207017089891,
    0.267138152891,
  };
  double normalizer = {
    1.0 / (muls[0] + muls[1]),
  };
  static const double r0 = 2.3770330432;
  static const double r1 = 9.04353323561;
  static const double r2 = 9.24456601467;
  static const double border_ratio = -0.0724948220913;

  {
    // X component
    ImageF diff = DiffPrecompute(xyb0[0], xyb1[0]);
    ImageF blurred = Blur(diff, r2, border_ratio);
    (*mask)[0] = ImageF(xsize, ysize);
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        (*mask)[0].Row(y)[x] = blurred.Row(y)[x];
      }
    }
  }
  {
    // Y component
    (*mask)[1] = ImageF(xsize, ysize);
    ImageF diff = DiffPrecompute(xyb0[1], xyb1[1]);
    ImageF blurred1 = Blur(diff, r0, border_ratio);
    ImageF blurred2 = Blur(diff, r1, border_ratio);
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        const double val = normalizer * (
            muls[0] * blurred1.Row(y)[x] +
            muls[1] * blurred2.Row(y)[x]);
        (*mask)[1].Row(y)[x] = val;
      }
    }
  }
  // B component
  (*mask)[2] = ImageF(xsize, ysize);
  static const double mul[2] = {
    16.6963293877,
    2.1364621982,
  };
  static const double w00 = 36.4671237619;
  static const double w11 = 2.1887170895;
  static const double w_ytob_hf = std::max<double>(
      0.086624184478,
      0.0);
  static const double w_ytob_lf = 21.6804277046;
  static const double p1_to_p0 = 0.0513061271723;

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
  static const double fuzzy_width_up = 6.07887388532;
  static const double fuzzy_width_down = 5.50793514384;
  static const double m0 = 2.0;
  static const double scaler = 0.840253347958;
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
