/*
* OpenCL/CUDA edition implementation of ButteraugliComparator.
*
* Author: strongtu@tencent.com
*         ianhuang@tencent.com
*         chriskzhou@tencent.com
*/
#include <algorithm>
#include <stdint.h>
#include <vector>
#include "utils.h"

#ifdef __USE_OPENCL__

using namespace std;

int g_idvec[10] = { 0 };
int g_sizevec[10] = { 0 };

int get_global_id(int dim) {
    return g_idvec[dim];
}
int get_global_size(int dim) {
    return g_sizevec[dim];
}

void set_global_id(int dim, int id){
    g_idvec[dim] = id;
}
void set_global_size(int dim, int size){
    g_sizevec[dim] = size;
}

#define __checkcl
#define abs(exper)    fabs((exper))
#include "clguetzli.h"
#include "clguetzli.cl"
#include "cuguetzli.h"
#include "ocu.h"

namespace guetzli
{
    ButteraugliComparatorEx::ButteraugliComparatorEx(const int width, const int height,
        const std::vector<uint8_t>* rgb,
        const float target_distance, ProcessStats* stats)
        : ButteraugliComparator(width, height, rgb, target_distance, stats)
    {
        if (MODE_CPU != g_mathMode)
        {
            rgb_orig_opsin.resize(3);
            rgb_orig_opsin[0].resize(width * height);
            rgb_orig_opsin[1].resize(width * height);
            rgb_orig_opsin[2].resize(width * height);

#ifdef __USE_DOUBLE_AS_FLOAT__
            const float* lut = kSrgb8ToLinearTable;
#else
            const double* lut = kSrgb8ToLinearTable;
#endif
            for (int c = 0; c < 3; ++c) {
                for (int y = 0, ix = 0; y < height_; ++y) {
                    for (int x = 0; x < width_; ++x, ++ix) {
                        rgb_orig_opsin[c][ix] = lut[rgb_orig_[3 * ix + c]];
                    }
                }
            }
            ::butteraugli::OpsinDynamicsImage(width_, height_, rgb_orig_opsin);
        }
    }

    void ButteraugliComparatorEx::Compare(const OutputImage& img)
    {
		if (MODE_CPU_OPT == g_mathMode)
		{
			std::vector<std::vector<float> > rgb0 = rgb_orig_opsin;

			std::vector<std::vector<float> > rgb(3, std::vector<float>(width_ * height_));
			img.ToLinearRGB(&rgb);
			::butteraugli::OpsinDynamicsImage(width_, height_, rgb);
			std::vector<float>().swap(distmap_);
			comparator_.DiffmapOpsinDynamicsImage(rgb0, rgb, distmap_);
			distance_ = ::butteraugli::ButteraugliScoreFromDiffmap(distmap_);
		}
#ifdef __USE_OPENCL__
        else if (MODE_OPENCL == g_mathMode)
        {
            std::vector<std::vector<float> > rgb1(3, std::vector<float>(width_ * height_));
            img.ToLinearRGB(&rgb1);

            const int xsize = width_;
            const int ysize = height_;
            std::vector<float>().swap(distmap_);
            distmap_.resize(xsize * ysize);

            size_t channel_size = xsize * ysize * sizeof(float);
            ocl_args_d_t &ocl = getOcl();
            ocl_channels xyb0 = ocl.allocMemChannels(channel_size, rgb_orig_opsin[0].data(), rgb_orig_opsin[1].data(), rgb_orig_opsin[2].data());
            ocl_channels xyb1 = ocl.allocMemChannels(channel_size, rgb1[0].data(), rgb1[1].data(), rgb1[2].data());

            cl_mem mem_result = ocl.allocMem(channel_size);

            clOpsinDynamicsImageEx(xyb1, xsize, ysize);
            clDiffmapOpsinDynamicsImageEx(mem_result, xyb0, xyb1, xsize, ysize, comparator_.step());

            cl_int err = clEnqueueReadBuffer(ocl.commandQueue, mem_result, false, 0, channel_size, distmap_.data(), 0, NULL, NULL);
            LOG_CL_RESULT(err);
            err = clFinish(ocl.commandQueue);
            LOG_CL_RESULT(err);

            clReleaseMemObject(mem_result);
            ocl.releaseMemChannels(xyb0);
            ocl.releaseMemChannels(xyb1);

            distance_ = ::butteraugli::ButteraugliScoreFromDiffmap(distmap_);
        }
#endif
#ifdef __USE_CUDA__
        else if (MODE_CUDA == g_mathMode)
        {
            std::vector<std::vector<float> > rgb1(3, std::vector<float>(width_ * height_));
            img.ToLinearRGB(&rgb1);

            const int xsize = width_;
            const int ysize = height_;
            std::vector<float>().swap(distmap_);
            distmap_.resize(xsize * ysize);

            size_t channel_size = xsize * ysize * sizeof(float);
            ocu_args_d_t &ocu = getOcu();
            ocu_channels xyb0 = ocu.allocMemChannels(channel_size, rgb_orig_opsin[0].data(), rgb_orig_opsin[1].data(), rgb_orig_opsin[2].data());
            ocu_channels xyb1 = ocu.allocMemChannels(channel_size, rgb1[0].data(), rgb1[1].data(), rgb1[2].data());
            
            cu_mem mem_result = ocu.allocMem(channel_size);

            cuOpsinDynamicsImageEx(xyb1, xsize, ysize);

            cuDiffmapOpsinDynamicsImageEx(mem_result, xyb0, xyb1, xsize, ysize, comparator_.step());

            cuMemcpyDtoH(distmap_.data(), mem_result, channel_size);

            ocu.releaseMem(mem_result);
            ocu.releaseMemChannels(xyb0);
            ocu.releaseMemChannels(xyb1);

            distance_ = ::butteraugli::ButteraugliScoreFromDiffmap(distmap_);
        }
#endif
		else
		{
			ButteraugliComparator::Compare(img);
		}
    }

    void ButteraugliComparatorEx::StartBlockComparisons()
    {
        if (MODE_CPU == g_mathMode)
        {
            ButteraugliComparator::StartBlockComparisons();
            return;
        }

        std::vector<std::vector<float> > dummy(3);
        ::butteraugli::Mask(rgb_orig_opsin, rgb_orig_opsin, width_, height_, &mask_xyz_, &dummy);

        const int width = width_;
        const int height = height_;
        const int factor_x = 1;
        const int factor_y = 1;

        const int block_width = (width + 8 * factor_x - 1) / (8 * factor_x);
        const int block_height = (height + 8 * factor_y - 1) / (8 * factor_y);
        const int num_blocks = block_width * block_height;
#ifdef __USE_DOUBLE_AS_FLOAT__
        const float* lut = kSrgb8ToLinearTable;
#else
        const double* lut = kSrgb8ToLinearTable;
#endif
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

                CalcOpsinDynamicsImage((float(*)[64])curR);

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
    
    double ButteraugliComparatorEx::CompareBlock(const OutputImage& img, int off_x, int off_y, const coeff_t* candidate_block, const int comp_mask) const
    {
        double err = ButteraugliComparator::CompareBlock(img, off_x, off_y, candidate_block, comp_mask);
        return err;
    }
}

#endif