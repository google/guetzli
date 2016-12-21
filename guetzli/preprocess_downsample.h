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

// Preprocesses U and V channel for better results after downsampling.

#ifndef GUETZLI_PREPROCESS_DOWNSAMPLE_H_
#define GUETZLI_PREPROCESS_DOWNSAMPLE_H_

#include <stdint.h>
#include <vector>

namespace guetzli {

// Preprocesses the u (1) or v (2) channel of the given YUV image (range 0-255).
std::vector<std::vector<float>> PreProcessChannel(
    int w, int h, int channel, float sigma, float amount, bool blur,
    bool sharpen, const std::vector<std::vector<float>>& image);

// Gamma-compensated chroma subsampling.
// Returns Y, U, V image planes, each with width x height dimensions, but the
// U and V planes are composed of 2x2 blocks with the same values.
std::vector<std::vector<float> > RGBToYUV420(
    const std::vector<uint8_t>& rgb_in, const int width, const int height);

}  // namespace guetzli

#endif  // GUETZLI_PREPROCESS_DOWNSAMPLE_H_
