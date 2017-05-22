#include <algorithm>
#include <stdint.h>
#include <vector>
#include "utils.h"

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

#define __opencl
#define abs(exper)    fabs((exper))
#include "clguetzli.h"
#include "clguetzli.cl"

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

        const double* lut = kSrgb8ToLinearTable;

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
        if (g_checkOpenCL)
        {
            channel_info mayout_channel[3];
            for (int c = 0; c < 3; c++)
            {
                mayout_channel[c].block_height = img.component(c).height_in_blocks();
                mayout_channel[c].block_width = img.component(c).width_in_blocks();
                mayout_channel[c].factor = img.component(c).factor_x();
                mayout_channel[c].pixel = img.component(c).pixels();
                mayout_channel[c].coeff = img.component(c).coeffs();
            }

            double err2 = CompareBlockFactor(mayout_channel,
                candidate_block,
                block_x_,
                block_y_,
                imgOpsinDynamicsBlockList.data(),
                imgMaskXyzScaleBlockList.data(),
                width_,
                height_,
                factor_x_);

            if (err != err2)
            {
                LogError("CompareBlock miss %s(%d) \r\n", __FUNCTION__, __LINE__);
            }
        }

        return err;
    }
}
