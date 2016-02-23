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

const double kButteraugliGood = 1.632;
const double kButteraugliBad = 2.095;

bool ButteraugliInterface(size_t xsize, size_t ysize,
                          const std::vector<std::vector<double> > &rgb0,
                          const std::vector<std::vector<double> > &rgb1,
                          std::vector<double> &diffmap,
                          double &diffvalue);

// Implementation details, don't use anything below or your code will
// break in the future.
void ButteraugliMap(
    size_t xsize, size_t ysize,
    const std::vector<std::vector<double> > &rgb0,
    const std::vector<std::vector<double> > &rgb1,
    std::vector<double> &diffmap);

double ButteraugliDistanceFromMap(
    size_t xsize, size_t ysize,
    const std::vector<double> &distmap);

// Only referenced by butteraugli_test
// Compute masking impact from the intensity values of a rgb image.
void IntensityMasking(const std::vector<std::vector<double> > &rgb,
                      size_t xsize, size_t ysize,
                      double *mask);

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

// Non-linearities for component-based suppression.
double SuppressionRedPlusGreen(double delta);
double SuppressionRedMinusGreen(double delta);
double SuppressionBlue(double delta);

// Compute values of local frequency masking based on the activity
// in the argb image.
void SuppressionRgb(const std::vector<std::vector<double> > &rgb,
                    const std::vector<std::vector<double> > &blurred,
                    size_t xsize, size_t ysize,
                    std::vector<std::vector<double> > *suppression);

}  // namespace butteraugli

#endif  // BUTTERAUGLI_BUTTERAUGLI_H_
