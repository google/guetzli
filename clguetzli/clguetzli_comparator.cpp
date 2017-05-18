#include <stdint.h>
#include <algorithm>
#include "clguetzli_comparator.h"
#include "guetzli\idct.h"
#include "guetzli\color_transform.h"
#include "guetzli\gamma_correct.h"
#include "clguetzli\ocl.h"
#include "clguetzli\clguetzli.h"

using namespace guetzli;

void CoeffToIDCT(const coeff_t block[8*8], uint8_t idct[8*8])
{
	guetzli::ComputeBlockIDCT(block, idct);
}

void IDCTToPixel8x8(const uint8_t idct[8 * 8], uint16_t pixels_[8*8])
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

void IDCTToPixel16x16(const uint8_t idct[8*8], uint16_t pixels_out[16*16], const uint16_t *pixel_orig, int block_x, int block_y, int width_, int height_)
{
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
            }
            else if (y0 < 0) {
                subsampled[ix] = subsampled[ix + kSubsampledEdgeSize];
            }
            else if (x0 >= width_) {
                subsampled[ix] = subsampled[ix - 1];
            }
            else if (y0 >= height_) {
                subsampled[ix] = subsampled[ix - kSubsampledEdgeSize];
            }
            else if (i < 8 && j < 8) {
                subsampled[ix] = idct[j * 8 + i] << 4;
            }
            else {
                // Reconstruct the subsampled pixels around the edge of the current
                // block by computing the inverse of the fancy upsampler.
                const int y1 = std::max(y0 - 1, 0);
                const int x1 = std::max(x0 - 1, 0);
                subsampled[ix] = (pixel_orig[y0 * width_ + x0] * 9 +
                    pixel_orig[y1 * width_ + x1] +
                    pixel_orig[y0 * width_ + x1] * -3 +
                    pixel_orig[y1 * width_ + x0] * -3) >> 2;
            }
        }
    }
	// Determine area to update.
    int xmin = block_x * 16; // std::max(block_x * 16 - 1, 0);
    int xmax = std::min(block_x * 16 + 15, width_ -  1);
    int ymin = block_y * 16; // std::max(block_y * 16 - 1, 0);
    int ymax = std::min(block_y * 16 + 15, height_ - 1);

    // Apply the fancy upsampler on the subsampled block.
    for (int y = ymin; y <= ymax; ++y) {
        const int y0 = ((y & ~1) / 2 - block_y * 8 + 1) * kSubsampledEdgeSize;
        const int dy = ((y & 1) * 2 - 1) * kSubsampledEdgeSize;
        for (int x = xmin; x <= xmax; ++x) {
            const int x0 = (x & ~1) / 2 - block_x * 8 + 1;
            const int dx = (x & 1) * 2 - 1;
            const int ix = x0 + y0;

            int out_x = x - xmin;
            int out_y = y - ymin;

            pixels_out[out_y * 16 + out_x] = (subsampled[ix] * 9 + subsampled[ix + dy] * 3 +
                subsampled[ix + dx] * 3 + subsampled[ix + dx + dy]) >> 4;
        }
    }
}

// out = [YUVYUV....YUVYUV]
void PixelToYUV(uint16_t pixels_[8*8], uint8_t out[8*8], int xsize = 8, int ysize = 8)
{
	const int stride = 3;

	for (int y = 0; y < xsize; ++y) {
		for (int x = 0; x < ysize; ++x) {
            int px = y * xsize + x;
			*out = static_cast<uint8_t>((pixels_[px] + 8 - (x & 1)) >> 4);
            out += stride;
		}
	}
}

// pixel = [YUVYUV...YUVYUV] to [RGBRGB...RGBRGB]
void YUVToRGB(uint8_t pixelBlock[3*8*8], int size = 8 * 8)
{
	for (int i = 0; i < size; i++)
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

void YUVToImage(uint8_t yuv[3 * 8 * 8], float* r, float* g, float* b, int xsize = 8, int ysize = 8, int inside_x = 8, int inside_y = 8)
{
    YUVToRGB(yuv, xsize * ysize);

    const double* lut = Srgb8ToLinearTable();

    for (int i = 0; i < xsize * ysize; i++)
    {
        r[i] = lut[yuv[3 * i]];
        g[i] = lut[yuv[3 * i + 1]];
        b[i] = lut[yuv[3 * i + 2]];
    }
    for (int y = 0; y < inside_y; y++)
    {
        for (int x = inside_x; x < xsize; x++)
        {
            int idx = y * xsize + (inside_x - 1);
            r[y * xsize + x] = r[idx];
            g[y * xsize + x] = g[idx];
            b[y * xsize + x] = b[idx];
        }
    }
    for (int y = inside_y; y < ysize; y++)
    {
        for (int x = 0; x < xsize; x++)
        {
            int idx = (inside_y - 1) * xsize + x;
            r[y * xsize + x] = r[idx];
            g[y * xsize + x] = g[idx];
            b[y * xsize + x] = b[idx];
        }
    }
}

// block = [R....R][G....G][B.....]
void BlockToImage(const coeff_t block[8*8*3], float* r, float* g, float* b, int inside_x, int inside_y)
{
	uint8_t idct[3][8 * 8];
	CoeffToIDCT(&block[0], idct[0]);
	CoeffToIDCT(&block[8 * 8], idct[1]);
	CoeffToIDCT(&block[8 * 8 * 2], idct[2]);

    uint16_t pixels[3][8 * 8];
	IDCTToPixel8x8(idct[0], pixels[0]);
	IDCTToPixel8x8(idct[1], pixels[1]);
	IDCTToPixel8x8(idct[2], pixels[2]);

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

void CoeffToYUV16x16(const coeff_t block[8 * 8], uint8_t *yuv, const uint16_t *pixel_orig, int block_x, int block_y, int width_, int height_)
{
    uint8_t idct[8 * 8];
    CoeffToIDCT(&block[0], &idct[0]);

    uint16_t pixels[16 * 16];
    IDCTToPixel16x16(idct, pixels, pixel_orig, block_x, block_y, width_, height_);

    PixelToYUV(pixels, yuv, 16, 16);
}

void CoeffToYUV8x8(const coeff_t block[8 * 8], uint8_t *yuv)
{
    uint8_t idct[8 * 8];
    CoeffToIDCT(&block[0], &idct[0]);

    uint16_t pixels[8 * 8];
    IDCTToPixel8x8(idct, pixels);

    PixelToYUV(pixels, yuv);
}

void Copy8x8To16x16(const uint8_t yuv8x8[3 * 8 * 8], uint8_t yuv16x16[3 * 16 * 16], int off_x, int off_y)
{
    for (int y = 0; y < 8; y++)
    {
        for (int x = 0; x < 8; x++)
        {
            int idx = y * 8 + x;
            int idx16 = (y + off_y * 8) * 16 + (x + off_x * 8);
            yuv16x16[idx16 * 3] = yuv8x8[idx * 3];
        }
    }
}

void Copy16x16To8x8(const uint8_t yuv16x16[3 * 16 * 16], uint8_t yuv8x8[3 * 8 * 8], int off_x, int off_y)
{
    for (int y = 0; y < 8; y++)
    {
        for (int x = 0; x < 8; x++)
        {
            int idx = y * 8 + x;
            int idx16 = (y + off_y * 8) * 16 + (x + off_x * 8);
            yuv8x8[idx * 3] = yuv16x16[idx16 * 3];
        }
    }
}

void Copy16x16ToChannel(const float rgb16x16[3][16 * 16], float r[8 * 8], float g[8 * 8], float b[8 * 8], int off_x, int off_y)
{
    for (int y = 0; y < 8; y++)
    {
        for (int x = 0; x < 8; x++)
        {
            int idx = y * 8 + x;
            int idx16 = (y + off_y * 8) * 16 + (x + off_x * 8);
            r[idx] = rgb16x16[0][idx16];
            g[idx] = rgb16x16[1][idx16];
            b[idx] = rgb16x16[2][idx16];
        }
    }
}

typedef struct __channel_info_t
{
    int factor;
    int block_width;
    int block_height;
    const uint16_t *pixel;
}channel_info;

void ComputeBlockFacor(const coeff_t* candidate_block,
                       const coeff_t * mayout_coeff[3],
                       const channel_info mayout_channel[3],
                       const coeff_t * orig_coeff[3],
                       const int comp_mask,
                       int factor
)
{

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

    double ButteraugliComparatorEx::CompareBlock(const OutputImage& img, int off_x, int off_y, const coeff_t* candidate_block, const int comp_mask) const
    {
        double err = CompareBlockEx2(img, off_x, off_y, candidate_block, comp_mask);
        if (g_checkOpenCL)
        {
            double err1 = ButteraugliComparator::CompareBlock(img, off_x, off_y, candidate_block, comp_mask);
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
        int inside_x = block_x_ * 8 + 8 > width_ ? width_ - block_x_ * 8 : 8;
        int inside_y = block_y_ * 8 + 8 > height_ ? height_ - block_y_ * 8 : 8;
        std::vector<std::vector<float> > rgb1_c(3, std::vector<float>(kDCTBlockSize));
        BlockToImage(candidate_block, rgb1_c[0].data(), rgb1_c[1].data(), rgb1_c[2].data(), inside_x, inside_y);
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
        return ComputeImage8x8Block(rgb0_c, rgb1_c, getCurrentBlock8x8Idx(off_x, off_y));
	}

    int ButteraugliComparatorEx::GetOrigBlock(std::vector< std::vector<float> > &rgb0_c, int off_x, int off_y) const
    {
        int block_xx = block_x_ * factor_x_ + off_x;
        int block_yy = block_y_ * factor_y_ + off_y;
        if (block_xx * 8 >= width_ || block_yy * 8 >= height_) return -1;

        const int block8_width = (width_ + 8 - 1) / 8;

        int block_ix = block_yy * block8_width + block_xx;

        rgb0_c.resize(3);
        const float*  block_opsin = &imgOpsinDynamicsBlockList[block_ix * 3 * kDCTBlockSize];
        for (int i = 0; i < 3; i++)
        {
            rgb0_c[i].resize(kDCTBlockSize);
            memcpy(rgb0_c[i].data(), block_opsin + i * kDCTBlockSize, kDCTBlockSize * sizeof(float));
        }

        return block_ix;
    }

    double ButteraugliComparatorEx::CompareBlockEx2(const OutputImage& img, int off_x, int off_y, const coeff_t* candidate_block, const int comp_mask) const
    {
        const int block_x = block_x_;
        const int block_y = block_y_;
        const int factor = factor_x_;

        const coeff_t *candidate_channel[3];
        channel_info mayout_channel[3];
        const coeff_t *mayout_coeff[3];
        for (int c = 0; c < 3; c++)
        {
            candidate_channel[c] = &candidate_block[c * 8 * 8];
            mayout_coeff[c] = img.component(c).coeffs();
            mayout_channel[c].block_height = img.component(c).height_in_blocks();
            mayout_channel[c].block_width  = img.component(c).width_in_blocks();
            mayout_channel[c].factor       = img.component(c).factor_x();
            mayout_channel[c].pixel =       img.component(c).pixels();
        }

        uint8_t yuv16x16[3 * 16 * 16] = { 0 };  // factor 2 mode output image
        uint8_t yuv8x8[3 * 8 * 8] = { 0 };      // factor 1 mode output image

        // 不管comp_mask如何，转换为RGB总是需要的
        for (int c = 0; c < 3; c++)
        {
            if (mayout_channel[c].factor == 1) {
                if (factor == 1) {  // channel_factor == factor 说明要介入运算，采用candidate中的系数
                    const coeff_t * coeff_block = candidate_channel[c];
                    CoeffToYUV8x8(coeff_block, &yuv8x8[c]);
                }
                else {
                    for (int iy = 0; iy < factor; ++iy) {
                        for (int ix = 0; ix < factor; ++ix) {
                            int block_xx = block_x * factor + ix;
                            int block_yy = block_y * factor + iy;

                            if (ix != off_x || iy != off_y) continue;
                            if (block_xx >= mayout_channel[c].block_width ||
                                block_yy >= mayout_channel[c].block_height)
                            {
                                continue;
                            }
                            int block_8x8idx = block_yy * mayout_channel[c].block_width + block_xx;
                            const coeff_t * coeff_block = mayout_coeff[c] + block_8x8idx * 8 * 8;
                            CoeffToYUV8x8(coeff_block, &yuv8x8[c]);

                            // copy YUV8x8 to YUV1616 corner
                            Copy8x8To16x16(&yuv8x8[c], &yuv16x16[c], ix, iy);
                        }
                    }
                }
            }
            else {
                if (factor == 1) {
                    int block_xx = block_x / mayout_channel[c].factor;
                    int block_yy = block_y / mayout_channel[c].factor;
                    int ix = block_x % mayout_channel[c].factor;;
                    int iy = block_y % mayout_channel[c].factor;

                    int block_16x16idx = block_yy * mayout_channel[c].block_width + block_xx;
                    const coeff_t * coeff_block = mayout_coeff[c] + block_16x16idx * 8 * 8;
/*
                    uint8_t ch[16 * 16] = { 0 };
                    img.component(c).ToPixels(block_xx * 8, block_yy * 8, 16, 16, ch, 1);
*/
                    CoeffToYUV16x16(coeff_block, &yuv16x16[c], mayout_channel[c].pixel, block_xx, block_yy, img.width(), img.height());

                    // copy YUV16x16 corner to YUV8x8
                    Copy16x16To8x8(&yuv16x16[c], &yuv8x8[c], ix, iy);
                }
                else {
                    const coeff_t * coeff_block = candidate_channel[c];
                    CoeffToYUV16x16(coeff_block, &yuv16x16[c], mayout_channel[c].pixel, block_x, block_y, img.width(), img.height());
                }
            }
        }

        if (factor == 1)
        {
            std::vector< std::vector<float> > rgb0_c;
            int block_8x8idx = GetOrigBlock(rgb0_c, 0, 0);
/*
            uint8_t yuv[3 * 8 * 8];

            std::vector<std::vector<float> > rgb1_c2(3, std::vector<float>(kDCTBlockSize));
            {
                int block_x = block_x_ * factor_x_ + off_x;
                int block_y = block_y_ * factor_y_ + off_y;
                int xmin = 8 * block_x;
                int ymin = 8 * block_y;

                img.ToLinearRGB(xmin, ymin, 8, 8, &rgb1_c2);

                img.component(0).ToPixels(xmin, ymin, 8, 8, &yuv[0], 3);
                img.component(1).ToPixels(xmin, ymin, 8, 8, &yuv[1], 3);
                img.component(2).ToPixels(xmin, ymin, 8, 8, &yuv[2], 3);
            }
*/
            int inside_x = block_x_ * 8 + 8 > width_ ? width_ - block_x_ * 8 : 8;
            int inside_y = block_y_ * 8 + 8 > height_ ? height_ - block_y_ * 8 : 8;
            std::vector<std::vector<float> > rgb1_c(3, std::vector<float>(kDCTBlockSize));
            YUVToImage(yuv8x8, rgb1_c[0].data(), rgb1_c[1].data(), rgb1_c[2].data(), 8, 8, inside_x, inside_y);
/*
            int count = 0;
            for (int i = 0; i < 64; i++)
            {
                if (rgb1_c[0][i] != rgb1_c2[0][i] ||
                    rgb1_c[1][i] != rgb1_c2[1][i] ||
                    rgb1_c[2][i] != rgb1_c2[2][i])
                {
                    count++;
                }
            }
            if (count > 0)
            {
                LogError("fdjskafjdlasfj");
            }
*/
            return ComputeImage8x8Block(rgb0_c, rgb1_c, block_8x8idx);
        }
        else
        {
            int inside_x = block_x_ * 16 + 16 > width_ ? width_ - block_x_ * 16 : 16;
            int inside_y = block_y_ * 16 + 16 > height_ ? height_ - block_y_ * 16 : 16;
/*
            uint8_t yuv[3 * 8 * 8];
            std::vector<std::vector<float> > rgb1_c2(3, std::vector<float>(kDCTBlockSize));
            {
                int block_x = block_x_ * factor_x_ + off_x;
                int block_y = block_y_ * factor_y_ + off_y;
                int xmin = 8 * block_x;
                int ymin = 8 * block_y;

                img.ToLinearRGB(xmin, ymin, 8, 8, &rgb1_c2);

                img.component(0).ToPixels(xmin, ymin, 8, 8, &yuv[0], 3);
                img.component(1).ToPixels(xmin, ymin, 8, 8, &yuv[1], 3);
                img.component(2).ToPixels(xmin, ymin, 8, 8, &yuv[2], 3);
            }

*/
            float rgb16x16[3][16 * 16];
            YUVToImage(yuv16x16, rgb16x16[0], rgb16x16[1], rgb16x16[2], 16, 16, inside_x, inside_y);

            std::vector< std::vector<float> > rgb0_c;
            int block_8x8idx = GetOrigBlock(rgb0_c, off_x, off_y);

            std::vector<std::vector<float> > rgb1_c(3, std::vector<float>(kDCTBlockSize));
            Copy16x16ToChannel(rgb16x16, rgb1_c[0].data(), rgb1_c[1].data(), rgb1_c[2].data(), off_x, off_y);

            return ComputeImage8x8Block(rgb0_c, rgb1_c, block_8x8idx);
        }
    }

    double ButteraugliComparatorEx::ComputeImage8x8Block(std::vector<std::vector<float> > &rgb0_c,
        std::vector<std::vector<float> > &rgb1_c,
        int block_8x8idx) const
    {
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
            diff += diff_xyz_dc[c] * imgMaskXyzScaleBlockList[block_8x8idx * 3 + c];
            diff += diff_xyz_ac[c] * imgMaskXyzScaleBlockList[block_8x8idx * 3 + c];
            diff_edge += diff_xyz_edge_dc[c] * imgMaskXyzScaleBlockList[block_8x8idx * 3 + c];
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

    int ButteraugliComparatorEx::getCurrentBlock8x8Idx(int off_x, int off_y) const
    {
        int block_xx = block_x_ * factor_x_ + off_x;
        int block_yy = block_y_ * factor_y_ + off_y;

        const int block8_width =  (width_ + 8 - 1) / 8;
        return block_yy * block8_width + block_xx;
    }
}
