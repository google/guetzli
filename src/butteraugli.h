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

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

// This is the main interface to butteraugli image similarity
// analysis function.

namespace butteraugli {

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

const double kButteraugliGood = 1.000;
const double kButteraugliBad = 1.088091;

bool ButteraugliInterface(size_t xsize, size_t ysize,
                          const std::vector<std::vector<float> > &rgb0,
                          const std::vector<std::vector<float> > &rgb1,
                          std::vector<float> &diffmap,
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

// Returns a map which can be used for adaptive quantization. Values can
// typically range from kButteraugliQuantLow to kButteraugliQuantHigh. Low
// values require coarse quantization (e.g. near random noise), high values
// require fine quantization (e.g. in smooth bright areas).
bool ButteraugliAdaptiveQuantization(size_t xsize, size_t ysize,
    const std::vector<std::vector<float> > &rgb, std::vector<float> &quant);

// Implementation details, don't use anything below or your code will
// break in the future.

// Allows incremental computation of the butteraugli map by keeping some
// intermediate results.
class ButteraugliComparator {
 public:
  ButteraugliComparator(size_t xsize, size_t ysize, int step);

  // Computes the butteraugli map from scratch, updates all intermediate
  // results.
  void DistanceMap(const std::vector<std::vector<float> > &rgb0,
                   const std::vector<std::vector<float> > &rgb1,
                   std::vector<float> &result);

  // Computes the butteraugli map by resuing some intermediate results from the
  // previous run.
  //
  // Must be called with the same rgb0 image as in the last DistanceMap() call.
  //
  // If changed[res_y * res_xsize_ + res_x] is false, it assumes that rgb1
  // did not change compared to the previous calls of this function or
  // of DistanceMap() anywhere within an 8x8 block with upper-left corner in
  // (step_ * res_x, step_ * res_y).
  void DistanceMapIncremental(const std::vector<std::vector<float> > &rgb0,
                              const std::vector<std::vector<float> > &rgb1,
                              const std::vector<bool>& changed,
                              std::vector<float> &result);

  // Copies the suppression map computed by the previous call to DistanceMap()
  // or DistanceMapIncremental() to *suppression.
  void GetSuppressionMap(std::vector<std::vector<float> >* suppression) {
    *suppression = scale_xyz_;
  }

 private:
  void Dct8x8mapIncremental(const std::vector<std::vector<float> > &rgb0,
                            const std::vector<std::vector<float> > &rgb1,
                            const std::vector<bool>& changed);

  void EdgeDetectorMap(const std::vector<std::vector<float> > &rgb0,
                       const std::vector<std::vector<float> > &rgb1);

  void SuppressionMap(const std::vector<std::vector<float> > &rgb0,
                      const std::vector<std::vector<float> > &rgb1);

  void CombineChannels(std::vector<float>* result);

  void FinalizeDistanceMap(std::vector<float>* result);

  const size_t xsize_;
  const size_t ysize_;
  const size_t num_pixels_;
  const int step_;
  const size_t res_xsize_;
  const size_t res_ysize_;

  // Contains the suppression map, 3 layers, each xsize_ * ysize_ in size.
  std::vector<std::vector<float> > scale_xyz_;
  // The blurred original used in the edge detector map.
  std::vector<std::vector<float> > blurred0_;
  // The following are all step_ x step_ subsampled maps containing
  // 3-dimensional vectors.
  std::vector<float> gamma_map_;
  std::vector<float> dct8x8map_dc_;
  std::vector<float> dct8x8map_ac_;
  std::vector<float> edge_detector_map_;
};

void ButteraugliMap(
    size_t xsize, size_t ysize,
    const std::vector<std::vector<float> > &rgb0,
    const std::vector<std::vector<float> > &rgb1,
    std::vector<float> &diffmap);

double ButteraugliDistanceFromMap(
    size_t xsize, size_t ysize,
    const std::vector<float> &distmap);

// Color difference evaluation for 'high frequency' color changes.
//
// Color difference is computed between two color pairs:
// (r0, g0, b0) and (r1, g1, b1).
//
// Values are expected to be between 0 and 255, but gamma corrected,
// i.e., a linear amount of photons is modulated with a fixed amount
// of change in these values.
double RgbDiff(double r0, double g0, double b0,
               double r1, double g1, double b1);

// Same as RgbDiff^2. Avoids unnecessary square root.
double RgbDiffSquared(double r0, double g0, double b0,
                      double r1, double g1, double b1);

double RgbDiffScaledSquared(double r0, double g0, double b0,
                            double r1, double g1, double b1,
                            const double scale[3]);

void RgbDiffSquaredMultiChannel(double r0, double g0, double b0, double *diff);

double RgbDiffLowFreq(double r0, double g0, double b0,
                      double r1, double g1, double b1);

// Same as RgbDiffLowFreq^2. Avoids unnecessary square root.
double RgbDiffLowFreqSquared(double r0, double g0, double b0,
                             double r1, double g1, double b1);

double RgbDiffLowFreqScaledSquared(double r0, double g0, double b0,
                                   double r1, double g1, double b1,
                                   const double scale[3]);

void RgbDiffSquaredXyzAccumulate(double r0, double g0, double b0,
                                 double r1, double g1, double b1,
                                 double factor, double res[3]);

void RgbDiffLowFreqSquaredXyzAccumulate(double r0, double g0, double b0,
                                        double r1, double g1, double b1,
                                        double factor, double res[3]);

// Version of rgb diff that applies gamma correction to the diffs.
// The rgb "background" values where the diffs occur are given as
// ave_r, ave_g, ave_b.
double RgbDiffGamma(double ave_r, double ave_g, double ave_b,
                    double r0, double g0, double b0,
                    double r1, double g1, double b1);

double RgbDiffGammaLowFreq(double ave_r, double ave_g, double ave_b,
                           double r0, double g0, double b0,
                           double r1, double g1, double b1);

// The high frequency color model used by RgbDiff().
static inline void RgbToXyz(double r, double g, double b,
                            double *valx, double *valy, double *valz) {
  static const double a0 = 0.19334520917582404;
  static const double a1 = -0.08311773494921797;
  static const double b0 = 0.07713792858953174;
  static const double b1 = 0.2208810782725995;
  static const double c0 = 0.26188332580170837;
  *valx = a0 * r + a1 * g;
  *valy = b0 * r + b1 * g;
  *valz = c0 * b;
}

// Non-linearities for component-based suppression.
double SuppressionRedPlusGreen(double delta);
double SuppressionRedMinusGreen(double delta);
double SuppressionBlue(double delta);

// Compute values of local frequency masking based on the activity
// in the argb image.
void SuppressionRgb(const std::vector<std::vector<float> > &rgb,
                    size_t xsize, size_t ysize,
                    std::vector<std::vector<float> > *suppression);

// The Dct computation used in butteraugli.
void ButteraugliDctd8x8(double m[64]);

// Rgbdiff for one 8x8 block.
void ButteraugliDctd8x8RgbDiff(const double gamma[3],
                               double rgb0[192],
                               double rgb1[192],
                               double diff_xyz_dc[3],
                               double diff_xyz_ac[3]);

double GammaDerivativeAvgMin(const double m0[64], const double m1[64]);

void MixGamma(double gamma[3]);

// Fills in coeffs[0..191] vector in such a way that if d[0..191] is the
// difference vector of the XYZ-color space DCT coefficients of an 8x8 block,
// then the butteraugli block error can be approximated with the
//   SUM(coeffs[i] * d[i]^2; i in 0..191)
// quadratic function.
void ButteraugliQuadraticBlockDiffCoeffsXyz(const double scale[3],
                                            const double gamma[3],
                                            const double rgb[192],
                                            double coeffs[192]);

void GaussBlurApproximation(size_t xsize, size_t ysize, float* channel,
                            double sigma);


}  // namespace butteraugli

#endif  // BUTTERAUGLI_BUTTERAUGLI_H_
