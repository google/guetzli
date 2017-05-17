#include <stdint.h>
#include <algorithm>
#include "clguetzli_comparator.h"
#include "guetzli\idct.h"
#include "guetzli\color_transform.h"
#include "guetzli\gamma_correct.h"
#include "clguetzli\ocl.h"
#include "clguetzli\clguetzli.h"

using namespace guetzli;

void CoeffToIDCT(const coeff_t *block, uint8_t *idct)
{
	guetzli::ComputeBlockIDCT(block, idct);
}

void IDCTToPixel(const uint8_t idct[8 * 8], uint16_t *pixels_)
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
void PixelToYUV(uint16_t *pixels_, uint8_t *out)
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
void BlockToImage(const coeff_t *block, float* r, float* g, float* b, int inside_x, int inside_y)
{
	uint8_t idct[3][8 * 8];
	CoeffToIDCT(&block[0], idct[0]);
	CoeffToIDCT(&block[8 * 8], idct[1]);
	CoeffToIDCT(&block[8 * 8 * 2], idct[2]);

    uint16_t pixels[3][8 * 8];

	IDCTToPixel(idct[0], pixels[0]);
	IDCTToPixel(idct[1], pixels[1]);
	IDCTToPixel(idct[2], pixels[2]);

	uint8_t yuv[8 * 8 * 3];

	PixelToYUV(pixels[0], &yuv[0]);
	PixelToYUV(pixels[1], &yuv[1]);
	PixelToYUV(pixels[2], &yuv[2]);

    YUVToRGB(yuv);

	const double* lut = Srgb8ToLinearTable();

	for (int i = 0; i < 8 * 8; i++)
	{
		r[i] = lut[yuv[3 * i]];
		g[i] = lut[yuv[3 * i + 1]];
		b[i] = lut[yuv[3 * i + 2]];
	}
    for (int y = 0; y < inside_y; y++)
    {
        for (int x = inside_x; x < 8; x++)
        {
            int idx = y * 8 + (inside_x - 1);
            r[y * 8 + x] = r[idx];
            g[y * 8 + x] = g[idx];
            b[y * 8 + x] = b[idx];
        }
    }
    for (int y = inside_y; y < 8; y++)
    {
        for (int x = 0; x < 8; x++)
        {
            int idx = (inside_y - 1) * 8 + x;
            r[y * 8 + x] = r[idx];
            g[y * 8 + x] = g[idx];
            b[y * 8 + x] = b[idx];
        }
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

    double ButteraugliComparatorEx::CompareBlock(const OutputImage& img, int off_x, int off_y, const coeff_t* candidate_block) const
    {
        double err = CompareBlockEx(img, off_x, off_y, candidate_block);
        if (g_checkOpenCL)
        {
            double err1 = ButteraugliComparator::CompareBlock(img, off_x, off_y, candidate_block);
            if (err1 != err)
            {
                LogError("CHK %s(%d) \r\n", __FUNCTION__, __LINE__);
            }
        }
       
        return err;
    }

    double ButteraugliComparatorEx::CompareBlockEx(const OutputImage& img, int off_x, int off_y, const coeff_t* candidate_block) const
    {
        int block_ix = getCurrentBlockIdx();

        const float*  block_opsin = &imgOpsinDynamicsBlockList[block_ix * 3 * kDCTBlockSize];

        // 这块是原始图像
        std::vector< std::vector<float> > rgb0_c;
        rgb0_c.resize(3);
        for (int i = 0; i < 3; i++)
        {
            rgb0_c[i].resize(kDCTBlockSize);
            memcpy(rgb0_c[i].data(), block_opsin + i * kDCTBlockSize, kDCTBlockSize * sizeof(float));
        }

        // img是全局优化后的图像，我们通过coeff_t数据反算出来rgb
        int border_x = block_x_ * 8 + 8 > width_ ? width_ - block_x_ * 8 : 8;
        int border_y = block_y_ * 8 + 8 > height_ ? height_ - block_y_ * 8 : 8;
        std::vector<std::vector<float> > rgb1_c(3, std::vector<float>(kDCTBlockSize));
        BlockToImage(candidate_block, rgb1_c[0].data(), rgb1_c[1].data(), rgb1_c[2].data(), border_x, border_y);
/*
        {
            // 可能还有问题，我们做一个校验
            int block_x = block_x_ * factor_x_ + off_x;
            int block_y = block_y_ * factor_y_ + off_y;
            int xmin = 8 * block_x;
            int ymin = 8 * block_y;

            std::vector<std::vector<float> > rgb1_c2(3, std::vector<float>(kDCTBlockSize));
            img.ToLinearRGB(xmin, ymin, 8, 8, &rgb1_c2);

            for (int i = 0; i < 3; i++)
            {
                for (int k = 0; k < 64; k++)
                {
                    if (fabs(rgb1_c[i][k] - rgb1_c2[i][k]) > 0.001)
                    {
                        LogError("Error: CompareBlock misstake.\n");
                    }
                }
            }
        }
*/
        // 下面是计算工作
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


    int ButteraugliComparatorEx::getCurrentBlockIdx(void) const
    {
        const int block_width = (width_ + 8 * factor_x_ - 1) / (8 * factor_x_);
        const int block_height = (height_ + 8 * factor_y_ - 1) / (8 * factor_y_);

        return block_y_ * block_width + block_x_;
    }
}
