#include <stdint.h>
#include <algorithm>
#include "clguetzli_comparator.h"
#include "guetzli\idct.h"
#include "guetzli\color_transform.h"
#include "guetzli\gamma_correct.h"

using namespace guetzli;

void CoeffToIDCT(coeff_t *block, uint8_t *idct)
{
	guetzli::ComputeBlockIDCT(block, idct);
}

void IDCTToImage(const uint8_t idct[8 * 8], uint16_t *pixels_)
{
	const int block_x = 0;
	const int block_y = 0;
	const int width_ = 8;
	const int height_ = 8;

	for (int iy = 0; iy < 8; ++iy) {
		for (int ix = 0; ix < 8; ++ix) {
			int x = 8 * block_x + ix;
			int y = 8 * block_y + iy;
			if (x >= width_ || y >= height_) continue;
			int p = y * width_ + x;
			pixels_[p] = idct[8 * iy + ix] << 4;
		}
	}
}

// out = [YUVYUV....YUVYUV]
void ImageToYUV(uint16_t *pixels_, uint8_t *out)
{
	const int stride = 3;

	for (int y = 0; y < 8; ++y) {
		for (int x = 0; x < 8; ++x) {
            int px = y * 8 + x;
			*out = static_cast<uint8_t>((pixels_[px] + 8 - (x & 1)) >> 4);
            out += stride;
		}
	}
}

// pixel = [YUVYUV...YUVYUV] to [RGBRGB...RGBRGB]
void YUVToRGB(uint8_t* pixelBlock)
{
	for (int i = 0; i < 64; i++)
	{
		uint8_t *pixel = &pixelBlock[i*3];

		int y = pixel[0];
		int cb = pixel[1];
		int cr = pixel[2];
		pixel[0] = kRangeLimit[y + kCrToRedTable[cr]];
		pixel[1] = kRangeLimit[y + ((kCrToGreenTable[cr] + kCbToGreenTable[cb]) >> 16)];
		pixel[2] = kRangeLimit[y + kCbToBlueTable[cb]];
	}
}

// block = [R....R][G....G][B.....]
void BlockToImage(coeff_t *block, float* r, float* g, float* b)
{
	uint8_t idct[8 * 8 * 3];
	CoeffToIDCT(&block[0], &idct[0]);
	CoeffToIDCT(&block[8 * 8], &idct[8 * 8]);
	CoeffToIDCT(&block[8 * 8 * 2], &idct[8 * 8 * 2]);

	uint16_t pixels[8 * 8 * 3];

	IDCTToImage(&idct[0], &pixels[0]);
	IDCTToImage(&idct[8*8], &pixels[8*8]);
	IDCTToImage(&idct[8*8*2], &pixels[8*8*2]);

	uint8_t yuv[8 * 8 * 3];

	ImageToYUV(&pixels[0], &yuv[0]);
	ImageToYUV(&pixels[8*8], &yuv[1]);
	ImageToYUV(&pixels[8*8*2], &yuv[2]);

    YUVToRGB(yuv);

	const double* lut = Srgb8ToLinearTable();
	for (int i = 0; i < 8 * 8; i++)
	{
		r[i] = lut[yuv[3 * i]];
		g[i] = lut[yuv[3 * i + 1]];
		b[i] = lut[yuv[3 * i + 2]];
	}
}

namespace guetzli
{
	ButteraugliComparatorEx::ButteraugliComparatorEx(const int width, const int height,
		const std::vector<uint8_t>* rgb,
		const float target_distance, ProcessStats* stats)
		: ButteraugliComparator(width, height, rgb, target_distance, stats)
	{

	}

	void ButteraugliComparatorEx::StartBlockComparisons()
	{
		ButteraugliComparator::StartBlockComparisons();

        const int width = width_;
        const int height = height_;
        const int factor_x = 1;
        const int factor_y = 1;

        const int block_width = (width + 8 * factor_x - 1) / (8 * factor_x);
        const int block_height = (height + 8 * factor_y - 1) / (8 * factor_y);
        const int num_blocks = block_width * block_height;
        
        const double* lut = Srgb8ToLinearTable();

        imgOpsinDynamicsBlockList.resize(num_blocks * 3 * kDCTBlockSize);
        imgMaskXyzScaleBlockList.resize(num_blocks * 3);
        for (int block_y = 0, block_ix = 0; block_y < block_height; ++block_y)
        {
            for (int block_x = 0; block_x < block_width; ++block_x, ++block_ix)
            {
                float* curR = &imgOpsinDynamicsBlockList[block_ix * 3 * kDCTBlockSize];
                float* curG = curR + kDCTBlockSize;
                float* curB = curG + kDCTBlockSize;

                for (int iy = 0, i = 0; iy < 8; ++iy) {
                    for (int ix = 0; ix < 8; ++ix, ++i) {
                        int x = std::min(8 * block_x + ix, width - 1);
                        int y = std::min(8 * block_y + iy, height - 1);
                        int px = y * width + x;

                        curR[i] = lut[rgb_orig_[3 * px]];
                        curG[i] = lut[rgb_orig_[3 * px + 1]];
                        curB[i] = lut[rgb_orig_[3 * px + 2]];
                    }
                }

                int xmin = block_x * 8;
                int ymin = block_y * 8;

                imgMaskXyzScaleBlockList[block_ix * 3] = mask_xyz_[0][ymin * width_ + xmin];
                imgMaskXyzScaleBlockList[block_ix * 3 + 1] = mask_xyz_[1][ymin * width_ + xmin];
                imgMaskXyzScaleBlockList[block_ix * 3 + 2] = mask_xyz_[2][ymin * width_ + xmin];
            }
        }
	}

    void ButteraugliComparatorEx::FinishBlockComparisons() {
        ButteraugliComparator::FinishBlockComparisons();

        imgOpsinDynamicsBlockList.clear();
        imgMaskXyzScaleBlockList.clear();
    }

	void ButteraugliComparatorEx::SwitchBlock(int block_x, int block_y, int factor_x, int factor_y)
	{
        block_x_ = block_x;
        block_y_ = block_y;
        factor_x_ = factor_x;
        factor_y_ = factor_y;

		ButteraugliComparator::SwitchBlock(block_x, block_y, factor_x, factor_y);
	}

	double ButteraugliComparatorEx::CompareBlockEx(coeff_t* candidate_block)
	{
        int block_ix = getCurrentBlockIdx();

        float*  block_opsin = &imgOpsinDynamicsBlockList[block_ix * 3 * kDCTBlockSize];

        // 这个内存拷贝待优化，但不是现在
        std::vector< std::vector<float> > rgb0_c;
        rgb0_c.resize(3);
        for (int i = 0; i < 3; i++)
        {
            rgb0_c[i].resize(kDCTBlockSize);
            memcpy(rgb0_c[i].data(), block_opsin + i*kDCTBlockSize, kDCTBlockSize * sizeof(float));
        }

        //
		std::vector<std::vector<float> > rgb1_c(3, std::vector<float>(kDCTBlockSize));
		BlockToImage(candidate_block, rgb1_c[0].data(), rgb1_c[1].data(), rgb1_c[2].data());

        ::butteraugli::OpsinDynamicsImage(8, 8, rgb0_c);
		::butteraugli::OpsinDynamicsImage(8, 8, rgb1_c);

		std::vector<std::vector<float> > rgb0 = rgb0_c;
		std::vector<std::vector<float> > rgb1 = rgb1_c;

		::butteraugli::MaskHighIntensityChange(8, 8, rgb0_c, rgb1_c, rgb0, rgb1);

		double b0[3 * kDCTBlockSize];
		double b1[3 * kDCTBlockSize];
		for (int c = 0; c < 3; ++c) {
			for (int ix = 0; ix < kDCTBlockSize; ++ix) {
				b0[c * kDCTBlockSize + ix] = rgb0[c][ix];
				b1[c * kDCTBlockSize + ix] = rgb1[c][ix];
			}
		}
		double diff_xyz_dc[3] = { 0.0 };
		double diff_xyz_ac[3] = { 0.0 };
		double diff_xyz_edge_dc[3] = { 0.0 };
		::butteraugli::ButteraugliBlockDiff(b0, b1, diff_xyz_dc, diff_xyz_ac, diff_xyz_edge_dc);

		double diff = 0.0;
		double diff_edge = 0.0;
		for (int c = 0; c < 3; ++c) {
            diff      += diff_xyz_dc[c]      * imgMaskXyzScaleBlockList[block_ix * 3 + c];
            diff      += diff_xyz_ac[c]      * imgMaskXyzScaleBlockList[block_ix * 3 + c];
            diff_edge += diff_xyz_edge_dc[c] * imgMaskXyzScaleBlockList[block_ix * 3 + c];
		}
        const double kEdgeWeight = 0.05;
		return sqrt((1 - kEdgeWeight) * diff + kEdgeWeight * diff_edge);
	}


    int ButteraugliComparatorEx::getCurrentBlockIdx(void)
    {
        const int width = width_;
        const int height = height_;
        const int factor_x = 1;
        const int factor_y = 1;

        const int block_width = (width + 8 * factor_x - 1) / (8 * factor_x);
        const int block_height = (height + 8 * factor_y - 1) / (8 * factor_y);

        return block_y_ * block_width + block_x_;
    }
}
