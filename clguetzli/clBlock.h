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

		void StartBlockComparisons();

		void SwitchBlock(int block_x, int block_y, int factor_x, int factor_y);

		double CompareBlockEx(const OutputImage& img, int off_x, int off_y, coeff_t* candidate_block);

	protected:
		std::vector<float> imgOpsinDynamicsBlockList;
	};

}