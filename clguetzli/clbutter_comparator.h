/*
* OpenCL/CUDA edition implementation of butter_comparator.
*
* Author: strongtu@tencent.com
*         ianhuang@tencent.com
*         chriskzhou@tencent.com
*/
#pragma once
#include <vector>
#include "butteraugli/butteraugli.h"

#define __restrict__

namespace butteraugli {

    class clButteraugliComparator : public ButteraugliComparator
    {
    public:
        clButteraugliComparator(size_t xsize, size_t ysize, int step);

        virtual void DiffmapOpsinDynamicsImage(std::vector<std::vector<float>> &xyb0,
            std::vector<std::vector<float>> &xyb1,
            std::vector<float> &result);

		virtual void DiffmapOpsinDynamicsImageOpt(std::vector<std::vector<float>> &xyb0,
			std::vector<std::vector<float>> &xyb1,
			std::vector<float> &result);

        virtual void BlockDiffMap(const std::vector<std::vector<float> > &rgb0,
            const std::vector<std::vector<float> > &rgb1,
            std::vector<float>* block_diff_dc,
            std::vector<float>* block_diff_ac);

		virtual void BlockDiffMapOpt(const std::vector<std::vector<float> > &rgb0,
			const std::vector<std::vector<float> > &rgb1,
			std::vector<float>* block_diff_dc,
			std::vector<float>* block_diff_ac);

        virtual void EdgeDetectorMap(const std::vector<std::vector<float> > &rgb0,
            const std::vector<std::vector<float> > &rgb1,
            std::vector<float>* edge_detector_map);

		virtual void EdgeDetectorMapOpt(const std::vector<std::vector<float> > &rgb0,
			const std::vector<std::vector<float> > &rgb1,
			std::vector<float>* edge_detector_map);

        virtual void EdgeDetectorLowFreq(const std::vector<std::vector<float> > &rgb0,
            const std::vector<std::vector<float> > &rgb1,
            std::vector<float>* block_diff_ac);

		virtual void EdgeDetectorLowFreqOpt(const std::vector<std::vector<float> > &rgb0,
			const std::vector<std::vector<float> > &rgb1,
			std::vector<float>* block_diff_ac);

        virtual void CombineChannels(const std::vector<std::vector<float> >& scale_xyb,
            const std::vector<std::vector<float> >& scale_xyb_dc,
            const std::vector<float>& block_diff_dc,
            const std::vector<float>& block_diff_ac,
            const std::vector<float>& edge_detector_map,
            std::vector<float>* result);

		virtual void CombineChannelsOpt(const std::vector<std::vector<float> >& scale_xyb,
			const std::vector<std::vector<float> >& scale_xyb_dc,
			const std::vector<float>& block_diff_dc,
			const std::vector<float>& block_diff_ac,
			const std::vector<float>& edge_detector_map,
			std::vector<float>* result);
    };

    void _MinSquareVal(size_t square_size, size_t offset, size_t xsize, size_t ysize, float *values);
    void _Average5x5(int xsize, int ysize, std::vector<float>* diffs);
    void _DiffPrecompute(const std::vector<std::vector<float> > &xyb0, const std::vector<std::vector<float> > &xyb1, size_t xsize, size_t ysize, std::vector<std::vector<float> > *mask);
    void _Mask(const std::vector<std::vector<float> > &xyb0,
        const std::vector<std::vector<float> > &xyb1,
        size_t xsize, size_t ysize,
        std::vector<std::vector<float> > *mask,
        std::vector<std::vector<float> > *mask_dc);
    void _CalculateDiffmap(const size_t xsize, const size_t ysize,
        const size_t step,
        std::vector<float>* diffmap);
    void _OpsinDynamicsImage(size_t xsize, size_t ysize,
        std::vector<std::vector<float> > &rgb);
    void _MaskHighIntensityChange(
        size_t xsize, size_t ysize,
        const std::vector<std::vector<float> > &c0,
        const std::vector<std::vector<float> > &c1,
        std::vector<std::vector<float> > &xyb0,
        std::vector<std::vector<float> > &xyb1);
    void _ScaleImage(double scale, std::vector<float> *result);
    void _Convolution(size_t xsize, size_t ysize,
        size_t xstep,
        size_t len, size_t offset,
        const float* __restrict__ multipliers,
        const float* __restrict__ inp,
        double border_ratio,
        float* __restrict__ result);
    void _Blur(size_t xsize, size_t ysize, float* channel, double sigma,
        double border_ratio);

    void MinSquareVal(size_t square_size, size_t offset, size_t xsize, size_t ysize, float *values);
    void Average5x5(int xsize, int ysize, std::vector<float>* diffs);
    void DiffPrecompute(const std::vector<std::vector<float> > &xyb0, const std::vector<std::vector<float> > &xyb1, size_t xsize, size_t ysize, std::vector<std::vector<float> > *mask);
    void ScaleImage(double scale, std::vector<float> *result);
    void Convolution(size_t xsize, size_t ysize,
        size_t xstep,
        size_t len, size_t offset,
        const float* __restrict__ multipliers,
        const float* __restrict__ inp,
        float border_ratio,
        float* __restrict__ result);
    void Blur(size_t xsize, size_t ysize, float* channel, double sigma,
        double border_ratio);
    void CalculateDiffmap(const size_t xsize, const size_t ysize,
        const size_t step,
        std::vector<float>* diffmap);
}