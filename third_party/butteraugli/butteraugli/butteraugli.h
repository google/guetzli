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

#define BUTTERAUGLI_ENABLE_CHECKS 0

// This is the main interface to butteraugli image similarity
// analysis function.

namespace butteraugli {

template<typename T>
class Image;

using Image8 = Image<uint8_t>;
using ImageF = Image<float>;
using ImageD = Image<double>;

// ButteraugliInterface defines the public interface for butteraugli.
//
// It calculates the difference between rgb0 and rgb1.
//
// rgb0 and rgb1 contain the images. rgb0[c][px] and rgb1[c][px] contains
// the red image for c == 0, green for c == 1, blue for c == 2. Location index
// px is calculated as y * xsize + x.
//
// Value of pixels of images rgb0 and rgb1 need to be represented as raw
// intensity. Most image formats store gamma corrected intensity in pixel
// values. This gamma correction has to be removed, by applying the following
// function:
// butteraugli_val = 255.0 * pow(png_val / 255.0, gamma);
// A typical value of gamma is 2.2. It is usually stored in the image header.
// Take care not to confuse that value with its inverse. The gamma value should
// be always greater than one.
// Butteraugli does not work as intended if the caller does not perform
// gamma correction.
//
// diffmap will contain an image of the size xsize * ysize, containing
// localized differences for values px (indexed with the px the same as rgb0
// and rgb1). diffvalue will give a global score of similarity.
//
// A diffvalue smaller than kButteraugliGood indicates that images can be
// observed as the same image.
// diffvalue larger than kButteraugliBad indicates that a difference between
// the images can be observed.
// A diffvalue between kButteraugliGood and kButteraugliBad indicates that
// a subtle difference can be observed between the images.
//
// Returns true on success.

bool ButteraugliInterface(const std::vector<ImageF> &rgb0,
                          const std::vector<ImageF> &rgb1,
                          ImageF &diffmap,
                          double &diffvalue);

const double kButteraugliQuantLow = 0.26;
const double kButteraugliQuantHigh = 1.454;

// Converts the butteraugli score into fuzzy class values that are continuous
// at the class boundary. The class boundary location is based on human
// raters, but the slope is arbitrary. Particularly, it does not reflect
// the expectation value of probabilities of the human raters. It is just
// expected that a smoother class boundary will allow for higher-level
// optimization algorithms to work faster.
//
// Returns 2.0 for a perfect match, and 1.0 for 'ok', 0.0 for bad. Because the
// scoring is fuzzy, a butteraugli score of 0.96 would return a class of
// around 1.9.
double ButteraugliFuzzyClass(double score);

// Input values should be in range 0 (bad) to 2 (good). Use
// kButteraugliNormalization as normalization.
double ButteraugliFuzzyInverse(double seek);

// Returns a map which can be used for adaptive quantization. Values can
// typically range from kButteraugliQuantLow to kButteraugliQuantHigh. Low
// values require coarse quantization (e.g. near random noise), high values
// require fine quantization (e.g. in smooth bright areas).
bool ButteraugliAdaptiveQuantization(size_t xsize, size_t ysize,
    const std::vector<std::vector<float> > &rgb, std::vector<float> &quant);

// Implementation details, don't use anything below or your code will
// break in the future.

#ifdef _MSC_VER
#define BUTTERAUGLI_RESTRICT
#else
#define BUTTERAUGLI_RESTRICT __restrict__
#endif

#ifdef _MSC_VER
#define BUTTERAUGLI_CACHE_ALIGNED_RETURN /* not supported */
#else
#define BUTTERAUGLI_CACHE_ALIGNED_RETURN __attribute__((assume_aligned(64)))
#endif

// Alias for unchangeable, non-aliased pointers. T is a pointer type,
// possibly to a const type. Example: ConstRestrict<uint8_t*> ptr = nullptr.
// The conventional syntax uint8_t* const RESTRICT is more confusing - it is
// not immediately obvious that the pointee is non-const.
template <typename T>
using ConstRestrict = T const BUTTERAUGLI_RESTRICT;

// Functions that depend on the cache line size.
class CacheAligned {
 public:
  static constexpr size_t kPointerSize = sizeof(void *);
  static constexpr size_t kCacheLineSize = 64;

  // The aligned-return annotation is only allowed on function declarations.
  static void *Allocate(const size_t bytes) BUTTERAUGLI_CACHE_ALIGNED_RETURN;
  static void Free(void *aligned_pointer);
};

template <typename T>
using CacheAlignedUniquePtrT = std::unique_ptr<T[], void (*)(void *)>;

using CacheAlignedUniquePtr = CacheAlignedUniquePtrT<uint8_t>;

template <typename T = uint8_t>
static inline CacheAlignedUniquePtrT<T> Allocate(const size_t entries) {
  return CacheAlignedUniquePtrT<T>(
      static_cast<ConstRestrict<T *>>(
          CacheAligned::Allocate(entries * sizeof(T))),
      CacheAligned::Free);
}

// Returns the smallest integer not less than "amount" that is divisible by
// "multiple", which must be a power of two.
template <size_t multiple>
static inline size_t Align(const size_t amount) {
  static_assert(multiple != 0 && ((multiple & (multiple - 1)) == 0),
                "Align<> argument must be a power of two");
  return (amount + multiple - 1) & ~(multiple - 1);
}

// Single channel, contiguous (cache-aligned) rows separated by padding.
// T must be POD.
//
// Rationale: vectorization benefits from aligned operands - unaligned loads and
// especially stores are expensive when the address crosses cache line
// boundaries. Introducing padding after each row ensures the start of a row is
// aligned, and that row loops can process entire vectors (writes to the padding
// are allowed and ignored).
//
// We prefer a planar representation, where channels are stored as separate
// 2D arrays, because that simplifies vectorization (repeating the same
// operation on multiple adjacent components) without the complexity of a
// hybrid layout (8 R, 8 G, 8 B, ...). In particular, clients can easily iterate
// over all components in a row and Image requires no knowledge of the pixel
// format beyond the component type "T". The downside is that we duplicate the
// xsize/ysize members for each channel.
//
// This image layout could also be achieved with a vector and a row accessor
// function, but a class wrapper with support for "deleter" allows wrapping
// existing memory allocated by clients without copying the pixels. It also
// provides convenient accessors for xsize/ysize, which shortens function
// argument lists. Supports move-construction so it can be stored in containers.
template <typename ComponentType>
class Image {
  // Returns cache-aligned row stride, being careful to avoid 2K aliasing.
  static size_t BytesPerRow(const size_t xsize) {
    // Allow reading one extra AVX-2 vector on the right margin.
    const size_t row_size = xsize * sizeof(T) + 32;
    const size_t align = CacheAligned::kCacheLineSize;
    size_t bytes_per_row = (row_size + align - 1) & ~(align - 1);
    // During the lengthy window before writes are committed to memory, CPUs
    // guard against read after write hazards by checking the address, but
    // only the lower 11 bits. We avoid a false dependency between writes to
    // consecutive rows by ensuring their sizes are not multiples of 2 KiB.
    if (bytes_per_row % 2048 == 0) {
      bytes_per_row += align;
    }
    return bytes_per_row;
  }

 public:
  using T = ComponentType;

  Image() : xsize_(0), ysize_(0), bytes_per_row_(0), bytes_(static_cast<uint8_t*>(nullptr), Ignore) {}

  Image(const size_t xsize, const size_t ysize)
      : xsize_(xsize),
        ysize_(ysize),
        bytes_per_row_(BytesPerRow(xsize)),
        bytes_(Allocate(bytes_per_row_ * ysize)) {}

  Image(const size_t xsize, const size_t ysize, ConstRestrict<uint8_t *> bytes,
        const size_t bytes_per_row)
      : xsize_(xsize),
        ysize_(ysize),
        bytes_per_row_(bytes_per_row),
        bytes_(bytes, Ignore) {}

  // Move constructor (required for returning Image from function)
  Image(Image &&other)
      : xsize_(other.xsize_),
        ysize_(other.ysize_),
        bytes_per_row_(other.bytes_per_row_),
        bytes_(std::move(other.bytes_)) {}

  // Move assignment (required for std::vector)
  Image &operator=(Image &&other) {
    xsize_ = other.xsize_;
    ysize_ = other.ysize_;
    bytes_per_row_ = other.bytes_per_row_;
    bytes_ = std::move(other.bytes_);
    return *this;
  }

  void Swap(Image &other) {
    std::swap(xsize_, other.xsize_);
    std::swap(ysize_, other.ysize_);
    std::swap(bytes_per_row_, other.bytes_per_row_);
    std::swap(bytes_, other.bytes_);
  }

  // How many pixels.
  size_t xsize() const { return xsize_; }
  size_t ysize() const { return ysize_; }

  ConstRestrict<T *> Row(const size_t y) BUTTERAUGLI_CACHE_ALIGNED_RETURN {
#ifdef BUTTERAUGLI_ENABLE_CHECKS
    if (y >= ysize_) {
      printf("Row %zu out of bounds (ysize=%zu)\n", y, ysize_);
      abort();
    }
#endif
    return reinterpret_cast<T *>(bytes_.get() + y * bytes_per_row_);
  }

  ConstRestrict<const T *> Row(const size_t y) const
      BUTTERAUGLI_CACHE_ALIGNED_RETURN {
#ifdef BUTTERAUGLI_ENABLE_CHECKS
    if (y >= ysize_) {
      printf("Const row %zu out of bounds (ysize=%zu)\n", y, ysize_);
      abort();
    }
#endif
    return reinterpret_cast<const T *>(bytes_.get() + y * bytes_per_row_);
  }

  // Raw access to byte contents, for interfacing with other libraries.
  // Unsigned char instead of char to avoid surprises (sign extension).
  ConstRestrict<uint8_t *> bytes() { return bytes_.get(); }
  ConstRestrict<const uint8_t *> bytes() const { return bytes_.get(); }
  size_t bytes_per_row() const { return bytes_per_row_; }

  // Returns number of pixels (some of which are padding) per row. Useful for
  // computing other rows via pointer arithmetic.
  intptr_t PixelsPerRow() const {
    static_assert(CacheAligned::kCacheLineSize % sizeof(T) == 0,
                  "Padding must be divisible by the pixel size.");
    return static_cast<intptr_t>(bytes_per_row_ / sizeof(T));
  }

 private:
  // Deleter used when bytes are not owned.
  static void Ignore(void *ptr) {}

  // (Members are non-const to enable assignment during move-assignment.)
  size_t xsize_;  // original intended pixels, not including any padding.
  size_t ysize_;
  size_t bytes_per_row_;  // [bytes] including padding.
  CacheAlignedUniquePtr bytes_;
};

// Returns newly allocated planes of the given dimensions.
template <typename T>
static inline std::vector<Image<T>> CreatePlanes(const size_t xsize,
                                                 const size_t ysize,
                                                 const size_t num_planes) {
  std::vector<Image<T>> planes;
  planes.reserve(num_planes);
  for (size_t i = 0; i < num_planes; ++i) {
    planes.emplace_back(xsize, ysize);
  }
  return planes;
}

// Returns a new image with the same dimensions and pixel values.
template <typename T>
static inline Image<T> CopyPixels(const Image<T> &other) {
  Image<T> copy(other.xsize(), other.ysize());
  const void *BUTTERAUGLI_RESTRICT from = other.bytes();
  void *BUTTERAUGLI_RESTRICT to = copy.bytes();
  memcpy(to, from, other.ysize() * other.bytes_per_row());
  return copy;
}

// Returns new planes with the same dimensions and pixel values.
template <typename T>
static inline std::vector<Image<T>> CopyPlanes(
    const std::vector<Image<T>> &planes) {
  std::vector<Image<T>> copy;
  copy.reserve(planes.size());
  for (const Image<T> &plane : planes) {
    copy.push_back(CopyPixels(plane));
  }
  return copy;
}

// Compacts a padded image into a preallocated packed vector.
template <typename T>
static inline void CopyToPacked(const Image<T> &from, std::vector<T> *to) {
  const size_t xsize = from.xsize();
  const size_t ysize = from.ysize();
#if BUTTERAUGLI_ENABLE_CHECKS
  if (to->size() < xsize * ysize) {
    printf("%zu x %zu exceeds %zu capacity\n", xsize, ysize, to->size());
    abort();
  }
#endif
  for (size_t y = 0; y < ysize; ++y) {
    ConstRestrict<const float*> row_from = from.Row(y);
    ConstRestrict<float*> row_to = to->data() + y * xsize;
    memcpy(row_to, row_from, xsize * sizeof(T));
  }
}

// Expands a packed vector into a preallocated padded image.
template <typename T>
static inline void CopyFromPacked(const std::vector<T> &from, Image<T> *to) {
  const size_t xsize = to->xsize();
  const size_t ysize = to->ysize();
  assert(from.size() == xsize * ysize);
  for (size_t y = 0; y < ysize; ++y) {
    ConstRestrict<const float*> row_from = from.data() + y * xsize;
    ConstRestrict<float*> row_to = to->Row(y);
    memcpy(row_to, row_from, xsize * sizeof(T));
  }
}

template <typename T>
static inline std::vector<Image<T>> PlanesFromPacked(
    const size_t xsize, const size_t ysize,
    const std::vector<std::vector<T>> &packed) {
  std::vector<Image<T>> planes;
  planes.reserve(packed.size());
  for (const std::vector<T> &p : packed) {
    planes.push_back(Image<T>(xsize, ysize));
    CopyFromPacked(p, &planes.back());
  }
  return planes;
}

template <typename T>
static inline std::vector<std::vector<T>> PackedFromPlanes(
    const std::vector<Image<T>> &planes) {
  assert(!planes.empty());
  const size_t num_pixels = planes[0].xsize() * planes[0].ysize();
  std::vector<std::vector<T>> packed;
  packed.reserve(planes.size());
  for (const Image<T> &image : planes) {
    packed.push_back(std::vector<T>(num_pixels));
    CopyToPacked(image, &packed.back());
  }
  return packed;
}

class ButteraugliComparator {
 public:
  ButteraugliComparator(size_t xsize, size_t ysize, int step);

  // Computes the butteraugli map between rgb0 and rgb1 and updates result.
  void Diffmap(const std::vector<ImageF> &rgb0,
               const std::vector<ImageF> &rgb1,
               ImageF &result);

  // Same as above, but OpsinDynamicsImage() was already applied to
  // rgb0 and rgb1.
  void DiffmapOpsinDynamicsImage(const std::vector<ImageF> &rgb0,
                                 const std::vector<ImageF> &rgb1,
                                 ImageF &result);

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

void ButteraugliDiffmap(const std::vector<ImageF> &rgb0,
                        const std::vector<ImageF> &rgb1,
                        ImageF &diffmap);

double ButteraugliScoreFromDiffmap(const ImageF& distmap);

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

void OpsinAbsorbance(const double in[3], double out[3]);

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

void RgbToXyb(double r, double g, double b,
              double *valx, double *valy, double *valz);

double SimpleGamma(double v);

double GammaMinArg();
double GammaMaxArg();

// Polynomial evaluation via Clenshaw's scheme (similar to Horner's).
// Template enables compile-time unrolling of the recursion, but must reside
// outside of a class due to the specialization.
template <int INDEX>
static inline void ClenshawRecursion(const double x, const double *coefficients,
                                     double *b1, double *b2) {
  const double x_b1 = x * (*b1);
  const double t = (x_b1 + x_b1) - (*b2) + coefficients[INDEX];
  *b2 = *b1;
  *b1 = t;

  ClenshawRecursion<INDEX - 1>(x, coefficients, b1, b2);
}

// Base case
template <>
inline void ClenshawRecursion<0>(const double x, const double *coefficients,
                                 double *b1, double *b2) {
  const double x_b1 = x * (*b1);
  // The final iteration differs - no 2 * x_b1 here.
  *b1 = x_b1 - (*b2) + coefficients[0];
}

// Rational polynomial := dividing two polynomial evaluations. These are easier
// to find than minimax polynomials.
struct RationalPolynomial {
  template <int N>
  static double EvaluatePolynomial(const double x,
                                   const double (&coefficients)[N]) {
    double b1 = 0.0;
    double b2 = 0.0;
    ClenshawRecursion<N - 1>(x, coefficients, &b1, &b2);
    return b1;
  }

  // Evaluates the polynomial at x (in [min_value, max_value]).
  inline double operator()(const float x) const {
    // First normalize to [0, 1].
    const double x01 = (x - min_value) / (max_value - min_value);
    // And then to [-1, 1] domain of Chebyshev polynomials.
    const double xc = 2.0 * x01 - 1.0;

    const double yp = EvaluatePolynomial(xc, p);
    const double yq = EvaluatePolynomial(xc, q);
    if (yq == 0.0) return 0.0;
    return static_cast<float>(yp / yq);
  }

  // Domain of the polynomials; they are undefined elsewhere.
  double min_value;
  double max_value;

  // Coefficients of T_n (Chebyshev polynomials of the first kind).
  // Degree 5/5 is a compromise between accuracy (0.1%) and numerical stability.
  double p[5 + 1];
  double q[5 + 1];
};

static inline float GammaPolynomial(float value) {
  // Generated by gamma_polynomial.m from equispaced x/gamma(x) samples.
  static const RationalPolynomial r = {
  0.770000000000000, 274.579999999999984,
  {
    881.979476556478289, 1496.058452015812463, 908.662212739659481,
    373.566100223287378, 85.840860336314364, 6.683258861509244,
  },
  {
    12.262350348616792, 20.557285797683576, 12.161463238367844,
    4.711532733641639, 0.899112889751053, 0.035662329617191,
  }};
  return r(value);
}

}  // namespace butteraugli

#endif  // BUTTERAUGLI_BUTTERAUGLI_H_
