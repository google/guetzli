#pragma once
#include <vector>
#include "guetzli\butteraugli_comparator.h"

namespace guetzli {

	class ButteraugliComparatorEx : public ButteraugliComparator
	{
	public:
		ButteraugliComparatorEx(const int width, const int height,
			const std::vector<uint8_t>* rgb,
			const float target_distance, ProcessStats* stats);

		void StartBlockComparisons() override;
        void FinishBlockComparisons() override;
		void SwitchBlock(int block_x, int block_y, int factor_x, int factor_y) override;

        double CompareBlock(const OutputImage& img, int off_x, int off_y, const coeff_t* candidate_block, const int comp_mask) const override;
		double CompareBlockEx(const OutputImage& img, int off_x, int off_y, const coeff_t* candidate_block) const;
        double CompareBlockEx2(const OutputImage& img, int off_x, int off_y, const coeff_t* candidate_block, const int comp_mask) const;
    private:
        int    GetOrigBlock(std::vector< std::vector<float> > &rgb0_c, int off_x, int off_y) const;
        double ComputeImage8x8Block(std::vector<std::vector<float> > &rgb0_c,
                                    std::vector<std::vector<float> > &rgb1_c,
                                    int block_8x8idx) const;

        int getCurrentBlockIdx(void) const;
        int getCurrentBlock8x8Idx(int off_x, int off_y) const;
	public:
		std::vector<float> imgOpsinDynamicsBlockList;   // [RR..RRGG..GGBB..BB]:blockCount
        std::vector<float> imgMaskXyzScaleBlockList;    // [RGBRGB..RGBRGB]:blockCount
	};

}