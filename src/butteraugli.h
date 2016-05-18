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
const double kButteraugliBad = 1.0563581223198708;

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
  static const double mul0 = 1.346049931681325;
  static const double mul1 = 1.2368889594842547;
  static const double a0 = mul0 * 0.171;
  static const double a1 = mul0 * -0.0812;
  static const double b0 = mul1 * 0.08265;
  static const double b1 = mul1 * 0.168;
  static const double c0 = 0.2710592310046686;
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

// Compute distance map based on differences in 8x8 dct coefficients.
void Dctd8x8mapWithRgbDiff(
    const std::vector<std::vector<float> > &rgb0,
    const std::vector<std::vector<float> > &rgb1,
    const std::vector<std::vector<float> > &scale_xyz,
    size_t xsize, size_t ysize,
    std::vector<float> *result);

// Rgbdiff for one 8x8 block.
double ButteraugliDctd8x8RgbDiff(const double scale[3],
                                 const double gamma[3],
                                 double rgb0[192],
                                 double rgb1[192]);

double GammaDerivativeAvgMin(const double m0[64], const double m1[64]);

// Makes a cluster of local errors to be more impactful than
// just a single error.
void ApplyErrorClustering(size_t xsize, size_t ysize,
                          std::vector<float>* distmap);

// Fills in coeffs[0..191] vector in such a way that if d[0..191] is the
// difference vector of the XYZ-color space DCT coefficients of an 8x8 block,
// then the butteraugli block error can be approximated with the
//   SUM(coeffs[i] * d[i]^2; i in 0..191)
// quadratic function.
void ButteraugliQuadraticBlockDiffCoeffsXyz(const double scale[3],
                                            const double gamma[3],
                                            const double rgb[192],
                                            double coeffs[192]);

}  // namespace butteraugli

#endif  // BUTTERAUGLI_BUTTERAUGLI_H_
