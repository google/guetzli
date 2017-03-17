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

#include "guetzli/output_image.h"

#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <cstdlib>

#include "guetzli/idct.h"
#include "guetzli/color_transform.h"
#include "guetzli/dct_double.h"
#include "guetzli/gamma_correct.h"
#include "guetzli/preprocess_downsample.h"
#include "guetzli/quantize.h"

namespace guetzli {

OutputImageComponent::OutputImageComponent(int w, int h)
    : width_(w), height_(h) {
  Reset(1, 1);
}

void OutputImageComponent::Reset(int factor_x, int factor_y) {
  factor_x_ = factor_x;
  factor_y_ = factor_y;
  width_in_blocks_ = (width_ + 8 * factor_x_ - 1) / (8 * factor_x_);
  height_in_blocks_ = (height_ + 8 * factor_y_ - 1) / (8 * factor_y_);
  num_blocks_ = width_in_blocks_ * height_in_blocks_;
  coeffs_ = std::vector<coeff_t>(num_blocks_ * kDCTBlockSize);
  pixels_ = std::vector<uint16_t>(width_ * height_, 128 << 4);
  for (int i = 0; i < kDCTBlockSize; ++i) quant_[i] = 1;
}

bool OutputImageComponent::IsAllZero() const {
  int numcoeffs = num_blocks_ * kDCTBlockSize;
  for (int i = 0; i < numcoeffs; ++i) {
    if (coeffs_[i] != 0) return false;
  }
  return true;
}

void OutputImageComponent::GetCoeffBlock(int block_x, int block_y,
                                         coeff_t block[kDCTBlockSize]) const {
  assert(block_x < width_in_blocks_);
  assert(block_y < height_in_blocks_);
  int offset = (block_y * width_in_blocks_ + block_x) * kDCTBlockSize;
  memcpy(block, &coeffs_[offset], kDCTBlockSize * sizeof(coeffs_[0]));
}

void OutputImageComponent::ToPixels(int xmin, int ymin, int xsize, int ysize,
                                    uint8_t* out, int stride) const {
  assert(xmin >= 0);
  assert(ymin >= 0);
  assert(xmin < width_);
  assert(ymin < height_);
  const int yend1 = ymin + ysize;
  const int yend0 = std::min(yend1, height_);
  int y = ymin;
  for (; y < yend0; ++y) {
    const int xend1 = xmin + xsize;
    const int xend0 = std::min(xend1, width_);
    int x = xmin;
    int px = y * width_ + xmin;
    for (; x < xend0; ++x, ++px, out += stride) {
      *out = static_cast<uint8_t>((pixels_[px] + 8 - (x & 1)) >> 4);
    }
    const int offset = -stride;
    for (; x < xend1; ++x) {
      *out = out[offset];
      out += stride;
    }
  }
  for (; y < yend1; ++y) {
    const int offset = -stride * xsize;
    for (int x = 0; x < xsize; ++x) {
      *out = out[offset];
      out += stride;
    }
  }
}

void OutputImageComponent::ToFloatPixels(float* out, int stride) const {
  assert(factor_x_ == 1);
  assert(factor_y_ == 1);
  for (int block_y = 0; block_y < height_in_blocks_; ++block_y) {
    for (int block_x = 0; block_x < width_in_blocks_; ++block_x) {
      coeff_t block[kDCTBlockSize];
      GetCoeffBlock(block_x, block_y, block);
      double blockd[kDCTBlockSize];
      for (int k = 0; k < kDCTBlockSize; ++k) {
        blockd[k] = block[k];
      }
      ComputeBlockIDCTDouble(blockd);
      for (int iy = 0; iy < 8; ++iy) {
        for (int ix = 0; ix < 8; ++ix) {
          int y = block_y * 8 + iy;
          int x = block_x * 8 + ix;
          if (y >= height_ || x >= width_) continue;
          out[(y * width_ + x) * stride] = blockd[8 * iy + ix] + 128.0;
        }
      }
    }
  }
}

void OutputImageComponent::SetCoeffBlock(int block_x, int block_y,
                                         const coeff_t block[kDCTBlockSize]) {
  assert(block_x < width_in_blocks_);
  assert(block_y < height_in_blocks_);
  int offset = (block_y * width_in_blocks_ + block_x) * kDCTBlockSize;
  memcpy(&coeffs_[offset], block, kDCTBlockSize * sizeof(coeffs_[0]));
  uint8_t idct[kDCTBlockSize];
  ComputeBlockIDCT(&coeffs_[offset], idct);
  UpdatePixelsForBlock(block_x, block_y, idct);
}

void OutputImageComponent::UpdatePixelsForBlock(
    int block_x, int block_y, const uint8_t idct[kDCTBlockSize]) {
  if (factor_x_ == 1 && factor_y_ == 1) {
    for (int iy = 0; iy < 8; ++iy) {
      for (int ix = 0; ix < 8; ++ix) {
        int x = 8 * block_x + ix;
        int y = 8 * block_y + iy;
        if (x >= width_ || y >= height_) continue;
        int p = y * width_ + x;
        pixels_[p] = idct[8 * iy + ix] << 4;
      }
    }
  } else if (factor_x_ == 2 && factor_y_ == 2) {
    // Fill in the 10x10 pixel area in the subsampled image that will be the
    // basis of the upsampling. This area is enough to hold the 3x3 kernel of
    // the fancy upsampler around each pixel.
    static const int kSubsampledEdgeSize = 10;
    uint16_t subsampled[kSubsampledEdgeSize * kSubsampledEdgeSize];
    for (int j = 0; j < kSubsampledEdgeSize; ++j) {
      // The order we fill in the rows is:
      //   8 rows intersecting the block, row below, row above
      const int y0 = block_y * 16 + (j < 9 ? j * 2 : -2);
      for (int i = 0; i < kSubsampledEdgeSize; ++i) {
        // The order we fill in each row is:
        //   8 pixels within the block, left edge, right edge
        const int ix = ((j < 9 ? (j + 1) * kSubsampledEdgeSize : 0) +
                        (i < 9 ? i + 1 : 0));
        const int x0 = block_x * 16 + (i < 9 ? i * 2 : -2);
        if (x0 < 0) {
          subsampled[ix] = subsampled[ix + 1];
        } else if (y0 < 0) {
          subsampled[ix] = subsampled[ix + kSubsampledEdgeSize];
        } else if (x0 >= width_) {
          subsampled[ix] = subsampled[ix - 1];
        } else if (y0 >= height_) {
          subsampled[ix] = subsampled[ix - kSubsampledEdgeSize];
        } else if (i < 8 && j < 8) {
          subsampled[ix] = idct[j * 8 + i] << 4;
        } else {
          // Reconstruct the subsampled pixels around the edge of the current
          // block by computing the inverse of the fancy upsampler.
          const int y1 = std::max(y0 - 1, 0);
          const int x1 = std::max(x0 - 1, 0);
          subsampled[ix] = (pixels_[y0 * width_ + x0] * 9 +
                            pixels_[y1 * width_ + x1] +
                            pixels_[y0 * width_ + x1] * -3 +
                            pixels_[y1 * width_ + x0] * -3) >> 2;
        }
      }
    }

    // Determine area to update.
    int xmin = std::max(block_x * 16 - 1, 0);
    int xmax = std::min(block_x * 16 + 16, width_ - 1);
    int ymin = std::max(block_y * 16 - 1, 0);
    int ymax = std::min(block_y * 16 + 16, height_ - 1);

    // Apply the fancy upsampler on the subsampled block.
    for (int y = ymin; y <= ymax; ++y) {
      const int y0 = ((y & ~1) / 2 - block_y * 8 + 1) * kSubsampledEdgeSize;
      const int dy = ((y & 1) * 2 - 1) * kSubsampledEdgeSize;
      uint16_t* rowptr = &pixels_[y * width_];
      for (int x = xmin; x <= xmax; ++x) {
        const int x0 = (x & ~1) / 2 - block_x * 8 + 1;
        const int dx = (x & 1) * 2 - 1;
        const int ix = x0 + y0;
        rowptr[x] = (subsampled[ix] * 9 + subsampled[ix + dy] * 3 +
                     subsampled[ix + dx] * 3 + subsampled[ix + dx + dy]) >> 4;
      }
    }
  } else {
    printf("Sampling ratio not supported: factor_x = %d factor_y = %d\n",
           factor_x_, factor_y_);
    exit(1);
  }
}

void OutputImageComponent::CopyFromJpegComponent(const JPEGComponent& comp,
                                                 int factor_x, int factor_y,
                                                 const int* quant) {
  Reset(factor_x, factor_y);
  assert(width_in_blocks_ <= comp.width_in_blocks);
  assert(height_in_blocks_ <= comp.height_in_blocks);
  const size_t src_row_size = comp.width_in_blocks * kDCTBlockSize;
  for (int block_y = 0; block_y < height_in_blocks_; ++block_y) {
    const coeff_t* src_coeffs = &comp.coeffs[block_y * src_row_size];
    for (int block_x = 0; block_x < width_in_blocks_; ++block_x) {
      coeff_t block[kDCTBlockSize];
      for (int i = 0; i < kDCTBlockSize; ++i) {
        block[i] = src_coeffs[i] * quant[i];
      }
      SetCoeffBlock(block_x, block_y, block);
      src_coeffs += kDCTBlockSize;
    }
  }
  memcpy(quant_, quant, sizeof(quant_));
}

void OutputImageComponent::ApplyGlobalQuantization(const int q[kDCTBlockSize]) {
  for (int block_y = 0; block_y < height_in_blocks_; ++block_y) {
    for (int block_x = 0; block_x < width_in_blocks_; ++block_x) {
      coeff_t block[kDCTBlockSize];
      GetCoeffBlock(block_x, block_y, block);
      if (QuantizeBlock(block, q)) {
        SetCoeffBlock(block_x, block_y, block);
      }
    }
  }
  memcpy(quant_, q, sizeof(quant_));
}

OutputImage::OutputImage(int w, int h)
    : width_(w),
      height_(h),
      components_(3, OutputImageComponent(w, h)) {}

void OutputImage::CopyFromJpegData(const JPEGData& jpg) {
  for (int i = 0; i < jpg.components.size(); ++i) {
    const JPEGComponent& comp = jpg.components[i];
    assert(jpg.max_h_samp_factor % comp.h_samp_factor == 0);
    assert(jpg.max_v_samp_factor % comp.v_samp_factor == 0);
    int factor_x = jpg.max_h_samp_factor / comp.h_samp_factor;
    int factor_y = jpg.max_v_samp_factor / comp.v_samp_factor;
    assert(comp.quant_idx < jpg.quant.size());
    components_[i].CopyFromJpegComponent(comp, factor_x, factor_y,
                                         &jpg.quant[comp.quant_idx].values[0]);
  }
}

namespace {

void SetDownsampledCoefficients(const std::vector<float>& pixels,
                                int factor_x, int factor_y,
                                OutputImageComponent* comp) {
  assert(pixels.size() == comp->width() * comp->height());
  comp->Reset(factor_x, factor_y);
  for (int block_y = 0; block_y < comp->height_in_blocks(); ++block_y) {
    for (int block_x = 0; block_x < comp->width_in_blocks(); ++block_x) {
      double blockd[kDCTBlockSize];
      int x0 = 8 * block_x * factor_x;
      int y0 = 8 * block_y * factor_y;
      assert(x0 < comp->width());
      assert(y0 < comp->height());
      for (int iy = 0; iy < 8; ++iy) {
        for (int ix = 0; ix < 8; ++ix) {
          float avg = 0.0;
          for (int j = 0; j < factor_y; ++j) {
            for (int i = 0; i < factor_x; ++i) {
              int x = std::min(x0 + ix * factor_x + i, comp->width() - 1);
              int y = std::min(y0 + iy * factor_y + j, comp->height() - 1);
              avg += pixels[y * comp->width() + x];
            }
          }
          avg /= factor_x * factor_y;
          blockd[iy * 8 + ix] = avg;
        }
      }
      ComputeBlockDCTDouble(blockd);
      blockd[0] -= 1024.0;
      coeff_t block[kDCTBlockSize];
      for (int k = 0; k < kDCTBlockSize; ++k) {
        block[k] = static_cast<coeff_t>(std::round(blockd[k]));
      }
      comp->SetCoeffBlock(block_x, block_y, block);
    }
  }
}

}  // namespace

void OutputImage::Downsample(const DownsampleConfig& cfg) {
  if (components_[1].IsAllZero() && components_[2].IsAllZero()) {
    // If the image is already grayscale, nothing to do.
    return;
  }
  if (cfg.use_silver_screen &&
      cfg.u_factor_x == 2 && cfg.u_factor_y == 2 &&
      cfg.v_factor_x == 2 && cfg.v_factor_y == 2) {
    std::vector<uint8_t> rgb = ToSRGB();
    std::vector<std::vector<float> > yuv = RGBToYUV420(rgb, width_, height_);
    SetDownsampledCoefficients(yuv[0], 1, 1, &components_[0]);
    SetDownsampledCoefficients(yuv[1], 2, 2, &components_[1]);
    SetDownsampledCoefficients(yuv[2], 2, 2, &components_[2]);
    return;
  }
  // Get the floating-point precision YUV array represented by the set of
  // DCT coefficients.
  std::vector<std::vector<float> > yuv(3, std::vector<float>(width_ * height_));
  for (int c = 0; c < 3; ++c) {
    components_[c].ToFloatPixels(&yuv[c][0], 1);
  }

  yuv = PreProcessChannel(width_, height_, 2, 1.3, 0.5,
                          cfg.u_sharpen, cfg.u_blur, yuv);
  yuv = PreProcessChannel(width_, height_, 1, 1.3, 0.5,
                          cfg.v_sharpen, cfg.v_blur, yuv);

  // Do the actual downsampling (averaging) and forward-DCT.
  if (cfg.u_factor_x != 1 || cfg.u_factor_y != 1) {
    SetDownsampledCoefficients(yuv[1], cfg.u_factor_x, cfg.u_factor_y,
                               &components_[1]);
  }
  if (cfg.v_factor_x != 1 || cfg.v_factor_y != 1) {
    SetDownsampledCoefficients(yuv[2], cfg.v_factor_x, cfg.v_factor_y,
                               &components_[2]);
  }
}

void OutputImage::ApplyGlobalQuantization(const int q[3][kDCTBlockSize]) {
  for (int c = 0; c < 3; ++c) {
    components_[c].ApplyGlobalQuantization(&q[c][0]);
  }
}

void OutputImage::SaveToJpegData(JPEGData* jpg) const {
  assert(components_[0].factor_x() == 1);
  assert(components_[0].factor_y() == 1);
  jpg->width = width_;
  jpg->height = height_;
  jpg->max_h_samp_factor = 1;
  jpg->max_v_samp_factor = 1;
  jpg->MCU_cols = components_[0].width_in_blocks();
  jpg->MCU_rows = components_[0].height_in_blocks();
  int ncomp = components_[1].IsAllZero() && components_[2].IsAllZero() ? 1 : 3;
  for (int i = 1; i < ncomp; ++i) {
    jpg->max_h_samp_factor = std::max(jpg->max_h_samp_factor,
                                      components_[i].factor_x());
    jpg->max_v_samp_factor = std::max(jpg->max_h_samp_factor,
                                      components_[i].factor_y());
    jpg->MCU_cols = std::min(jpg->MCU_cols, components_[i].width_in_blocks());
    jpg->MCU_rows = std::min(jpg->MCU_rows, components_[i].height_in_blocks());
  }
  jpg->components.resize(ncomp);
  int q[3][kDCTBlockSize];
  for (int c = 0; c < 3; ++c) {
    memcpy(&q[c][0], components_[c].quant(), kDCTBlockSize * sizeof(q[0][0]));
  }
  for (int c = 0; c < ncomp; ++c) {
    JPEGComponent* comp = &jpg->components[c];
    assert(jpg->max_h_samp_factor % components_[c].factor_x() == 0);
    assert(jpg->max_v_samp_factor % components_[c].factor_y() == 0);
    comp->id = c;
    comp->h_samp_factor = jpg->max_h_samp_factor / components_[c].factor_x();
    comp->v_samp_factor = jpg->max_v_samp_factor / components_[c].factor_y();
    comp->width_in_blocks = jpg->MCU_cols * comp->h_samp_factor;
    comp->height_in_blocks = jpg->MCU_rows * comp->v_samp_factor;
    comp->num_blocks = comp->width_in_blocks * comp->height_in_blocks;
    comp->coeffs.resize(kDCTBlockSize * comp->num_blocks);

    int last_dc = 0;
    const coeff_t* src_coeffs = components_[c].coeffs();
    coeff_t* dest_coeffs = &comp->coeffs[0];
    for (int block_y = 0; block_y < comp->height_in_blocks; ++block_y) {
      for (int block_x = 0; block_x < comp->width_in_blocks; ++block_x) {
        if (block_y >= components_[c].height_in_blocks() ||
            block_x >= components_[c].width_in_blocks()) {
          dest_coeffs[0] = last_dc;
          for (int k = 1; k < kDCTBlockSize; ++k) {
            dest_coeffs[k] = 0;
          }
        } else {
          for (int k = 0; k < kDCTBlockSize; ++k) {
            const int quant = q[c][k];
            int coeff = src_coeffs[k];
            assert(coeff % quant == 0);
            dest_coeffs[k] = coeff / quant;
          }
          src_coeffs += kDCTBlockSize;
        }
        last_dc = dest_coeffs[0];
        dest_coeffs += kDCTBlockSize;
      }
    }
  }
  SaveQuantTables(q, jpg);
}

std::vector<uint8_t> OutputImage::ToSRGB(int xmin, int ymin,
                                         int xsize, int ysize) const {
  std::vector<uint8_t> rgb(xsize * ysize * 3);
  for (int c = 0; c < 3; ++c) {
    components_[c].ToPixels(xmin, ymin, xsize, ysize, &rgb[c], 3);
  }
  for (int p = 0; p < rgb.size(); p += 3) {
    ColorTransformYCbCrToRGB(&rgb[p]);
  }
  return rgb;
}

std::vector<uint8_t> OutputImage::ToSRGB() const {
  return ToSRGB(0, 0, width_, height_);
}

void OutputImage::ToLinearRGB(int xmin, int ymin, int xsize, int ysize,
                              std::vector<std::vector<float> >* rgb) const {
  const double* lut = Srgb8ToLinearTable();
  std::vector<uint8_t> rgb_pixels = ToSRGB(xmin, ymin, xsize, ysize);
  for (int p = 0; p < xsize * ysize; ++p) {
    for (int i = 0; i < 3; ++i) {
      (*rgb)[i][p] = lut[rgb_pixels[3 * p + i]];
    }
  }
}

void OutputImage::ToLinearRGB(std::vector<std::vector<float> >* rgb) const {
  ToLinearRGB(0, 0, width_, height_, rgb);
}

std::string OutputImage::FrameTypeStr() const {
  char buf[128];
  int len = snprintf(buf, sizeof(buf), "f%d%d%d%d%d%d",
                     component(0).factor_x(), component(0).factor_y(),
                     component(1).factor_x(), component(1).factor_y(),
                     component(2).factor_x(), component(2).factor_y());
  return std::string(buf, len);
}

}  // namespace guetzli
