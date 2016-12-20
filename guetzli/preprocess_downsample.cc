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

#include "guetzli/preprocess_downsample.h"

#include <algorithm>
#include <assert.h>
#include <string.h>
#include <cmath>

using std::size_t;

namespace {

// convolve with size*size kernel
std::vector<float> Convolve2D(const std::vector<float>& image, int w, int h,
                              const double* kernel, int size) {
  auto result = image;
  int size2 = size / 2;
  for (int i = 0; i < image.size(); i++) {
    int x = i % w;
    int y = i / w;
    // Avoid non-normalized results at boundary by skipping edges.
    if (x < size2 || x + size - size2 - 1 >= w
        || y < size2 || y + size - size2 - 1 >= h) {
      continue;
    }
    float v = 0;
    for (int j = 0; j < size * size; j++) {
      int x2 = x + j % size - size2;
      int y2 = y + j / size - size2;
      v += kernel[j] * image[y2 * w + x2];
    }
    result[i] = v;
  }
  return result;
}

// convolve horizontally and vertically with 1D kernel
std::vector<float> Convolve2X(const std::vector<float>& image, int w, int h,
                              const double* kernel, int size, double mul) {
  auto temp = image;
  int size2 = size / 2;
  for (int i = 0; i < image.size(); i++) {
    int x = i % w;
    int y = i / w;
    // Avoid non-normalized results at boundary by skipping edges.
    if (x < size2 || x + size - size2 - 1 >= w) continue;
    float v = 0;
    for (int j = 0; j < size; j++) {
      int x2 = x + j - size2;
      v += kernel[j] * image[y * w + x2];
    }
    temp[i] = v * mul;
  }
  auto result = temp;
  for (int i = 0; i < temp.size(); i++) {
    int x = i % w;
    int y = i / w;
    // Avoid non-normalized results at boundary by skipping edges.
    if (y < size2 || y + size - size2 - 1 >= h) continue;
    float v = 0;
    for (int j = 0; j < size; j++) {
      int y2 = y + j - size2;
      v += kernel[j] * temp[y2 * w + x];
    }
    result[i] = v * mul;
  }
  return result;
}

double Normal(double x, double sigma) {
  static const double kInvSqrt2Pi = 0.3989422804014327;
  return std::exp(-x * x / (2 * sigma * sigma)) * kInvSqrt2Pi / sigma;
}

std::vector<float> Sharpen(const std::vector<float>& image, int w, int h,
                           float sigma, float amount) {
  // This is only made for small sigma, e.g. 1.3.
  std::vector<double> kernel(5);
  for (int i = 0; i < kernel.size(); i++) {
    kernel[i] = Normal(1.0 * i - kernel.size() / 2, sigma);
  }

  double sum = 0;
  for (int i = 0; i < kernel.size(); i++) sum += kernel[i];
  const double mul = 1.0 / sum;

  std::vector<float> result =
      Convolve2X(image, w, h, kernel.data(), kernel.size(), mul);
  for (size_t i = 0; i < image.size(); i++) {
    result[i] = image[i] + (image[i] - result[i]) * amount;
  }
  return result;
}

void Erode(int w, int h, std::vector<bool>* image) {
  std::vector<bool> temp = *image;
  for (int y = 1; y + 1 < h; y++) {
    for (int x = 1; x + 1 < w; x++) {
      size_t index = y * w + x;
      if (!(temp[index] && temp[index - 1] && temp[index + 1]
          && temp[index - w] && temp[index + w])) {
        (*image)[index] = 0;
      }
    }
  }
}

void Dilate(int w, int h, std::vector<bool>* image) {
  std::vector<bool> temp = *image;
  for (int y = 1; y + 1 < h; y++) {
    for (int x = 1; x + 1 < w; x++) {
      size_t index = y * w + x;
      if (temp[index] || temp[index - 1] || temp[index + 1]
          || temp[index - w] || temp[index + w]) {
        (*image)[index] = 1;
      }
    }
  }
}

std::vector<float> Blur(const std::vector<float>& image, int w, int h) {
    // This is only made for small sigma, e.g. 1.3.
    static const double kSigma = 1.3;
    std::vector<double> kernel(5);
    for (int i = 0; i < kernel.size(); i++) {
      kernel[i] = Normal(1.0 * i - kernel.size() / 2, kSigma);
    }

    double sum = 0;
    for (int i = 0; i < kernel.size(); i++) sum += kernel[i];
    const double mul = 1.0 / sum;

    return Convolve2X(image, w, h, kernel.data(), kernel.size(), mul);
}

}  // namespace

namespace guetzli {

// Do the sharpening to the v channel, but only in areas where it will help
// channel should be 2 for v sharpening, or 1 for less effective u sharpening
std::vector<std::vector<float>> PreProcessChannel(
    int w, int h, int channel, float sigma, float amount, bool blur,
    bool sharpen, const std::vector<std::vector<float>>& image) {
  if (!blur && !sharpen) return image;

  // Bring in range 0.0-1.0 for Y, -0.5 - 0.5 for U and V
  auto yuv = image;
  for (int i = 0; i < yuv[0].size(); i++) {
    yuv[0][i] /= 255.0;
    yuv[1][i] = yuv[1][i] / 255.0 - 0.5;
    yuv[2][i] = yuv[2][i] / 255.0 - 0.5;
  }

  // Map of areas where the image is not too bright to apply the effect.
  std::vector<bool> darkmap(image[0].size(), false);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      size_t index = y * w + x;
      float y = yuv[0][index];
      float u = yuv[1][index];
      float v = yuv[2][index];

      float r = y + 1.402 * v;
      float g = y - 0.34414 * u - 0.71414 * v;
      float b = y + 1.772 * u;

      // Parameters tuned to avoid sharpening in too bright areas, where the
      // effect makes it worse instead of better.
      if (channel == 2 && g < 0.85 && b < 0.85 && r < 0.9) {
        darkmap[index] = true;
      }
      if (channel == 1 && r < 0.85 && g < 0.85 && b < 0.9) {
        darkmap[index] = true;
      }
    }
  }

  Erode(w, h, &darkmap);
  Erode(w, h, &darkmap);
  Erode(w, h, &darkmap);

  // Map of areas where the image is red enough (blue in case of u channel).
  std::vector<bool> redmap(image[0].size(), false);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      size_t index = y * w + x;
      float u = yuv[1][index];
      float v = yuv[2][index];

      // Parameters tuned to allow only colors on which sharpening is useful.
      if (channel == 2 && 2.116 * v > -0.34414 * u + 0.2
          && 1.402 * v > 1.772 * u + 0.2) {
        redmap[index] = true;
      }
      if (channel == 1 && v < 1.263 * u - 0.1 && u > -0.33741 * v) {
        redmap[index] = true;
      }
    }
  }

  Dilate(w, h, &redmap);
  Dilate(w, h, &redmap);
  Dilate(w, h, &redmap);

  // Map of areas where to allow sharpening by combining red and dark areas
  std::vector<bool> sharpenmap(image[0].size(), 0);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      size_t index = y * w + x;
      sharpenmap[index] = redmap[index] && darkmap[index];
    }
  }

  // Threshold for where considered an edge.
  const double threshold = (channel == 2 ? 0.02 : 1.0) * 127.5;

  static const double kEdgeMatrix[9] = {
    0, -1, 0,
    -1, 4, -1,
    0, -1, 0
  };

  // Map of areas where to allow blurring, only where it is not too sharp
  std::vector<bool> blurmap(image[0].size(), false);
  std::vector<float> edge = Convolve2D(yuv[channel], w, h, kEdgeMatrix, 3);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      size_t index = y * w + x;
      float u = yuv[1][index];
      float v = yuv[2][index];
      if (sharpenmap[index]) continue;
      if (!darkmap[index]) continue;
      if (fabs(edge[index]) < threshold && v < -0.162 * u) {
        blurmap[index] = true;
      }
    }
  }
  Erode(w, h, &blurmap);
  Erode(w, h, &blurmap);

  // Choose sharpened, blurred or original per pixel
  std::vector<float> sharpened = Sharpen(yuv[channel], w, h, sigma, amount);
  std::vector<float> blurred = Blur(yuv[channel], w, h);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      size_t index = y * w + x;

      if (sharpenmap[index] > 0) {
        if (sharpen) yuv[channel][index] = sharpened[index];
      } else if (blurmap[index] > 0) {
        if (blur) yuv[channel][index] = blurred[index];
      }
    }
  }

  // Bring back to range 0-255
  for (int i = 0; i < yuv[0].size(); i++) {
    yuv[0][i] *= 255.0;
    yuv[1][i] = (yuv[1][i] + 0.5) * 255.0;
    yuv[2][i] = (yuv[2][i] + 0.5) * 255.0;
  }
  return yuv;
}

namespace {

inline float Clip(float val) {
  return std::max(0.0f, std::min(255.0f, val));
}

inline float RGBToY(float r, float g, float b) {
  return 0.299f * r + 0.587f * g + 0.114f * b;
}

inline float RGBToU(float r, float g, float b) {
  return -0.16874f * r - 0.33126f * g + 0.5f * b + 128.0;
}

inline float RGBToV(float r, float g, float b) {
  return 0.5f * r - 0.41869f * g - 0.08131f * b + 128.0;
}

inline float YUVToR(float y, float u, float v) {
  return y + 1.402 * (v - 128.0);
}

inline float YUVToG(float y, float u, float v) {
  return y - 0.344136 * (u - 128.0) - 0.714136 * (v - 128.0);
}

inline float YUVToB(float y, float u, float v) {
  return y + 1.772 * (u - 128.0);
}

// TODO(user) Use SRGB->linear conversion and a lookup-table.
inline float GammaToLinear(float x) {
  return std::pow(x / 255.0, 2.2);
}

// TODO(user) Use linear->SRGB conversion and a lookup-table.
inline float LinearToGamma(float x) {
  return 255.0 * std::pow(x, 1.0 / 2.2);
}

std::vector<float> LinearlyAveragedLuma(const std::vector<float>& rgb) {
  assert(rgb.size() % 3 == 0);
  std::vector<float> y(rgb.size() / 3);
  for (int i = 0, p = 0; p < rgb.size(); ++i, p += 3) {
    y[i] = LinearToGamma(RGBToY(GammaToLinear(rgb[p + 0]),
                                GammaToLinear(rgb[p + 1]),
                                GammaToLinear(rgb[p + 2])));
  }
  return y;
}

std::vector<float> LinearlyDownsample2x2(const std::vector<float>& rgb_in,
                                         const int width, const int height) {
  assert(rgb_in.size() == 3 * width * height);
  int w = (width + 1) / 2;
  int h = (height + 1) / 2;
  std::vector<float> rgb_out(3 * w * h);
  for (int y = 0, p = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int i = 0; i < 3; ++i, ++p) {
        rgb_out[p] = 0.0;
        for (int iy = 0; iy < 2; ++iy) {
          for (int ix = 0; ix < 2; ++ix) {
            int yy = std::min(height - 1, 2 * y + iy);
            int xx = std::min(width - 1, 2 * x + ix);
            rgb_out[p] += GammaToLinear(rgb_in[3 * (yy * width + xx) + i]);
          }
        }
        rgb_out[p] = LinearToGamma(0.25 * rgb_out[p]);
      }
    }
  }
  return rgb_out;
}

std::vector<std::vector<float> > RGBToYUV(const std::vector<float>& rgb) {
  std::vector<std::vector<float> > yuv(3, std::vector<float>(rgb.size() / 3));
  for (int i = 0, p = 0; p < rgb.size(); ++i, p += 3) {
    const float r = rgb[p + 0];
    const float g = rgb[p + 1];
    const float b = rgb[p + 2];
    yuv[0][i] = RGBToY(r, g, b);
    yuv[1][i] = RGBToU(r, g, b);
    yuv[2][i] = RGBToV(r, g, b);
  }
  return yuv;
}

std::vector<float> YUVToRGB(const std::vector<std::vector<float> >& yuv) {
  std::vector<float> rgb(3 * yuv[0].size());
  for (int i = 0, p = 0; p < rgb.size(); ++i, p += 3) {
    const float y = yuv[0][i];
    const float u = yuv[1][i];
    const float v = yuv[2][i];
    rgb[p + 0] = Clip(YUVToR(y, u, v));
    rgb[p + 1] = Clip(YUVToG(y, u, v));
    rgb[p + 2] = Clip(YUVToB(y, u, v));
  }
  return rgb;
}

// Upsamples img_in with a box-filter, and returns an image with output
// dimensions width x height.
std::vector<float> Upsample2x2(const std::vector<float>& img_in,
                               const int width, const int height) {
  int w = (width + 1) / 2;
  int h = (height + 1) / 2;
  assert(img_in.size() == w * h);
  std::vector<float> img_out(width * height);
  for (int y = 0, p = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x, ++p) {
      for (int iy = 0; iy < 2; ++iy) {
        for (int ix = 0; ix < 2; ++ix) {
          int yy = std::min(height - 1, 2 * y + iy);
          int xx = std::min(width - 1, 2 * x + ix);
          img_out[yy * width + xx] = img_in[p];
        }
      }
    }
  }
  return img_out;
}

// Apply the "fancy upsample" filter used by libjpeg.
std::vector<float> Blur(const std::vector<float>& img,
                        const int width, const int height) {
  std::vector<float> img_out(width * height);
  for (int y0 = 0; y0 < height; y0 += 2) {
    for (int x0 = 0; x0 < width; x0 += 2) {
      for (int iy = 0; iy < 2 && y0 + iy < height; ++iy) {
        for (int ix = 0; ix < 2 && x0 + ix < width; ++ix) {
          int dy = 4 * iy - 2;
          int dx = 4 * ix - 2;
          int x1 = std::min(width - 1, std::max(0, x0 + dx));
          int y1 = std::min(height - 1, std::max(0, y0 + dy));
          img_out[(y0 + iy) * width + x0 + ix] =
              (9.0 * img[y0 * width + x0] +
               3.0 * img[y0 * width + x1] +
               3.0 * img[y1 * width + x0] +
               1.0 * img[y1 * width + x1]) / 16.0;
        }
      }
    }
  }
  return img_out;
}

std::vector<float> YUV420ToRGB(const std::vector<std::vector<float> >& yuv420,
                               const int width, const int height) {
  std::vector<std::vector<float> > yuv;
  yuv.push_back(yuv420[0]);
  std::vector<float> u = Upsample2x2(yuv420[1], width, height);
  std::vector<float> v = Upsample2x2(yuv420[2], width, height);
  yuv.push_back(Blur(u, width, height));
  yuv.push_back(Blur(v, width, height));
  return YUVToRGB(yuv);
}

void UpdateGuess(const std::vector<float>& target,
                 const std::vector<float>& reconstructed,
                 std::vector<float>* guess) {
  assert(reconstructed.size() == guess->size());
  assert(target.size() == guess->size());
  for (int i = 0; i < guess->size(); ++i) {
    // TODO(user): Evaluate using a decaying constant here.
    (*guess)[i] = Clip((*guess)[i] - (reconstructed[i] - target[i]));
  }
}

}  // namespace

std::vector<std::vector<float> > RGBToYUV420(
    const std::vector<uint8_t>& rgb_in, const int width, const int height) {
  std::vector<float> rgbf(rgb_in.size());
  for (int i = 0; i < rgb_in.size(); ++i) {
    rgbf[i] = static_cast<float>(rgb_in[i]);
  }
  std::vector<float> y_target = LinearlyAveragedLuma(rgbf);
  std::vector<std::vector<float> > yuv_target =
      RGBToYUV(LinearlyDownsample2x2(rgbf, width, height));
  std::vector<std::vector<float> > yuv_guess = yuv_target;
  yuv_guess[0] = Upsample2x2(yuv_guess[0], width, height);
  // TODO(user): Stop early if the error is small enough.
  for (int iter = 0; iter < 20; ++iter) {
    std::vector<float> rgb_rec = YUV420ToRGB(yuv_guess, width, height);
    std::vector<float> y_rec = LinearlyAveragedLuma(rgb_rec);
    std::vector<std::vector<float> > yuv_rec =
        RGBToYUV(LinearlyDownsample2x2(rgb_rec, width, height));
    UpdateGuess(y_target, y_rec, &yuv_guess[0]);
    UpdateGuess(yuv_target[1], yuv_rec[1], &yuv_guess[1]);
    UpdateGuess(yuv_target[2], yuv_rec[2], &yuv_guess[2]);
  }
  yuv_guess[1] = Upsample2x2(yuv_guess[1], width, height);
  yuv_guess[2] = Upsample2x2(yuv_guess[2], width, height);
  return yuv_guess;
}

}  // namespace guetzli
