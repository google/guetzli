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

		double CompareBlockEx(coeff_t* candidate_block);
    private:
        int getCurrentBlockIdx(void);
	protected:
		std::vector<float> imgOpsinDynamicsBlockList;   // [RR..RRGG..GGBB..BB]:blockCount
        std::vector<float> imgMaskXyzScaleBlockList;    // [RGBRGB..RGBRGB]:blockCount
	};

}