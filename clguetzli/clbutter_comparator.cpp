#include "clbutter_comparator.h"
#include "clguetzli.h"
#include "clguetzli_test.h"

namespace butteraugli
{
    clButteraugliComparator::clButteraugliComparator(size_t xsize, size_t ysize, int step)
        : ButteraugliComparator(xsize, ysize, step)
    {

    }

    void clButteraugliComparator::DiffmapOpsinDynamicsImage(
        std::vector<std::vector<float>> &xyb0,
        std::vector<std::vector<float>> &xyb1,
        std::vector<float> &result)
    {
        if (MODE_OPENCL == g_mathMode && xsize_ > 100 && ysize_ > 100)
        {
            result.resize(xsize_ * ysize_);
            clDiffmapOpsinDynamicsImage(result.data(), xyb0[0].data(), xyb0[1].data(), xyb0[2].data(),
                xyb1[0].data(), xyb1[1].data(), xyb1[2].data(), xsize_, ysize_, step_);
        }
#ifdef __USE_CUDA__
        else if (MODE_CUDA == g_mathMode && xsize_ > 100 && ysize_ > 100)
        {
            result.resize(xsize_ * ysize_);
            cuDiffmapOpsinDynamicsImage(result.data(), xyb0[0].data(), xyb0[1].data(), xyb0[2].data(),
                xyb1[0].data(), xyb1[1].data(), xyb1[2].data(), xsize_, ysize_, step_);
        }
#endif
        else
        {
            ButteraugliComparator::DiffmapOpsinDynamicsImage(xyb0, xyb1, result);
        }
    }

    void clButteraugliComparator::BlockDiffMap(const std::vector<std::vector<float> > &xyb0,
        const std::vector<std::vector<float> > &xyb1,
        std::vector<float>* block_diff_dc,
        std::vector<float>* block_diff_ac)
    {
        ButteraugliComparator::BlockDiffMap(xyb0, xyb1, block_diff_dc, block_diff_ac);

        if (MODE_CHECKCL == g_mathMode && xsize_ > 8 && ysize_ > 8)
        {
            tclBlockDiffMap(xyb0[0].data(), xyb0[1].data(), xyb0[2].data(),
                xyb1[0].data(), xyb1[1].data(), xyb1[2].data(),
                xsize_, ysize_, step_,
                (*block_diff_dc).data(), (*block_diff_ac).data());
        }
    }


    void clButteraugliComparator::EdgeDetectorMap(const std::vector<std::vector<float> > &xyb0,
        const std::vector<std::vector<float> > &xyb1,
        std::vector<float>* edge_detector_map)
    {
        ButteraugliComparator::EdgeDetectorMap(xyb0, xyb1, edge_detector_map);

        if (MODE_CHECKCL == g_mathMode && xsize_ > 8 && ysize_ > 8)
        {
            tclEdgeDetectorMap(xyb0[0].data(), xyb0[1].data(), xyb0[2].data(),
                xyb1[0].data(), xyb1[1].data(), xyb1[2].data(),
                xsize_, ysize_, step_, 
                (*edge_detector_map).data());
        }
    }

    void clButteraugliComparator::EdgeDetectorLowFreq(const std::vector<std::vector<float> > &xyb0,
        const std::vector<std::vector<float> > &xyb1,
        std::vector<float>* block_diff_ac)
    {
        if (MODE_CHECKCL == g_mathMode && xsize_ > 8 && ysize_ > 8)
        {
            std::vector<float> orign_ac = *block_diff_ac;
            ButteraugliComparator::EdgeDetectorLowFreq(xyb0, xyb1, block_diff_ac);
            tclEdgeDetectorLowFreq(xyb0[0].data(), xyb0[1].data(), xyb0[2].data(),
                xyb1[0].data(), xyb1[1].data(), xyb1[2].data(),
                xsize_, ysize_, step_,
                orign_ac.data(), (*block_diff_ac).data());
        }
        else
        {
            ButteraugliComparator::EdgeDetectorLowFreq(xyb0, xyb1, block_diff_ac);
        }
    }

    void clButteraugliComparator::CombineChannels(const std::vector<std::vector<float> >& mask_xyb,
        const std::vector<std::vector<float> >& mask_xyb_dc,
        const std::vector<float>& block_diff_dc,
        const std::vector<float>& block_diff_ac,
        const std::vector<float>& edge_detector_map,
        std::vector<float>* result)
    {
        if (MODE_CHECKCL == g_mathMode && xsize_ > 8 && ysize_ > 8)
        {
            std::vector<float> temp = *result;
			temp.resize(res_xsize_ * res_ysize_);
            ButteraugliComparator::CombineChannels(mask_xyb, mask_xyb_dc, block_diff_dc, block_diff_ac, edge_detector_map, result);
            tclCombineChannels(mask_xyb[0].data(), mask_xyb[1].data(), mask_xyb[2].data(),
                mask_xyb_dc[0].data(), mask_xyb_dc[1].data(), mask_xyb_dc[2].data(),
                block_diff_dc.data(),
                block_diff_ac.data(), edge_detector_map.data(), xsize_, ysize_, res_xsize_, res_ysize_, step_, &temp[0], &(*result)[0]);
        }
        else
        {
            ButteraugliComparator::CombineChannels(mask_xyb, mask_xyb_dc, block_diff_dc, block_diff_ac, edge_detector_map, result);
        }
    }

    void MinSquareVal(size_t square_size, size_t offset, size_t xsize, size_t ysize, float *values) 
    {
        if (MODE_CHECKCL == g_mathMode && xsize > 8 && ysize > 8)
        {
            std::vector<float> img;
            img.resize(xsize * ysize);
            memcpy(img.data(), values, xsize * ysize * sizeof(float));
            _MinSquareVal(square_size, offset, xsize, ysize, values);
            tclMinSquareVal(img.data(), square_size, offset, xsize, ysize, values);
        }
        else
        {
            _MinSquareVal(square_size, offset, xsize, ysize, values);
        }
    }

    void Average5x5(int xsize, int ysize, std::vector<float>* diffs)
    {
        if (MODE_CHECKCL == g_mathMode && xsize > 8 && ysize > 8)
        {
            std::vector<float> diffs_org = *diffs;
            _Average5x5(xsize, ysize, diffs);
            tclAverage5x5(xsize, ysize, diffs_org, *diffs);
        }
        else
        {
            _Average5x5(xsize, ysize, diffs);
        }
    }

    void DiffPrecompute(const std::vector<std::vector<float> > &xyb0, const std::vector<std::vector<float> > &xyb1, size_t xsize, size_t ysize, std::vector<std::vector<float> > *mask)
    {
        _DiffPrecompute(xyb0, xyb1, xsize, ysize, mask);

        if (MODE_CHECKCL == g_mathMode && xsize > 8 && ysize > 8)
        {
            tclDiffPrecompute(xyb0, xyb1, xsize, ysize, mask);
        }
    }

    void Mask(const std::vector<std::vector<float> > &xyb0,
        const std::vector<std::vector<float> > &xyb1,
        size_t xsize, size_t ysize,
        std::vector<std::vector<float> > *mask,
        std::vector<std::vector<float> > *mask_dc)
    {
        if (MODE_OPENCL == g_mathMode && xsize > 100 && ysize > 100)
        {
            mask->resize(3);
            mask_dc->resize(3);
            for (int i = 0; i < 3; i++)
            {
                (*mask)[i].resize(xsize * ysize);
                (*mask_dc)[i].resize(xsize * ysize);
            }
            clMask((*mask)[0].data(), (*mask)[1].data(), (*mask)[2].data(),
                (*mask_dc)[0].data(), (*mask_dc)[1].data(), (*mask_dc)[2].data(),
                xsize, ysize,
                xyb0[0].data(), xyb0[1].data(), xyb0[2].data(),
                xyb1[0].data(), xyb1[1].data(), xyb1[2].data()
                );
        }
#ifdef __USE_CUDA__
        else if (MODE_CUDA == g_mathMode && xsize > 100 && ysize > 100)
        {
            mask->resize(3);
            mask_dc->resize(3);
            for (int i = 0; i < 3; i++)
            {
                (*mask)[i].resize(xsize * ysize);
                (*mask_dc)[i].resize(xsize * ysize);
            }
            cuMask((*mask)[0].data(), (*mask)[1].data(), (*mask)[2].data(),
                (*mask_dc)[0].data(), (*mask_dc)[1].data(), (*mask_dc)[2].data(),
                xsize, ysize,
                xyb0[0].data(), xyb0[1].data(), xyb0[2].data(),
                xyb1[0].data(), xyb1[1].data(), xyb1[2].data()
            );
        }
#endif
        else if (MODE_CHECKCL == g_mathMode && xsize > 8 && ysize > 8)
        {
            _Mask(xyb0, xyb1, xsize, ysize, mask, mask_dc);
            tclMask(xyb0[0].data(), xyb0[1].data(), xyb0[2].data(),
                xyb1[0].data(), xyb1[1].data(), xyb1[2].data(),
                xsize, ysize,
                (*mask)[0].data(), (*mask)[1].data(), (*mask)[2].data(),
                (*mask_dc)[0].data(), (*mask_dc)[1].data(), (*mask_dc)[2].data());
        }
        else
        {
            _Mask(xyb0, xyb1, xsize, ysize, mask, mask_dc);
        }
    }

    void CalculateDiffmap(const size_t xsize, const size_t ysize,
        const size_t step,
        std::vector<float>* diffmap)
    {
        if (MODE_CHECKCL == g_mathMode && xsize > 8 && ysize > 8)
        {
            std::vector<float> diffmap_org = *diffmap;
            _CalculateDiffmap(xsize, ysize, step, diffmap);
            tclCalculateDiffmap(xsize, ysize, step, diffmap_org.data(), diffmap_org.size(), (*diffmap).data());
        }
        else
        {
            _CalculateDiffmap(xsize, ysize, step, diffmap);
        }
    }

    void MaskHighIntensityChange(
        size_t xsize, size_t ysize,
        const std::vector<std::vector<float> > &c0,
        const std::vector<std::vector<float> > &c1,
        std::vector<std::vector<float> > &xyb0,
        std::vector<std::vector<float> > &xyb1)
    {
        _MaskHighIntensityChange(xsize, ysize, c0, c1, xyb0, xyb1);

        if (MODE_CHECKCL == g_mathMode && xsize > 8 && ysize > 8)
        {
            tclMaskHighIntensityChange(c0[0].data(), c0[1].data(), c0[2].data(),
                c1[0].data(), c1[1].data(), c1[2].data(),
                xsize, ysize,
                xyb0[0].data(), xyb0[1].data(), xyb0[2].data(),
                xyb1[0].data(), xyb1[1].data(), xyb1[2].data());
        }
    }

    void ScaleImage(double scale, std::vector<float> *result)
    {
        if (MODE_CHECKCL == g_mathMode && result->size() > 64)
        {
            std::vector<float> result_org = *result;
            _ScaleImage(scale, result);
            tclScaleImage(scale, result_org.data(), (*result).data(), (*result).size());
        }
        else
        {
            _ScaleImage(scale, result);
        }
    }

    void Convolution(size_t xsize, size_t ysize,
        size_t xstep,
        size_t len, size_t offset,
        const float* __restrict__ multipliers,
        const float* __restrict__ inp,
        float border_ratio,
        float* __restrict__ result)
    {
        _Convolution(xsize, ysize, xstep, len, offset, multipliers, inp, border_ratio, result);

        if (MODE_CHECKCL == g_mathMode && xsize > 8 && ysize > 8)
        {
            tclConvolution(xsize, ysize, xstep, len, offset, multipliers, inp, border_ratio, result);
        }
    }

    void Blur(size_t xsize, size_t ysize, float* channel, double sigma,
        double border_ratio)
    {
        if (MODE_CHECKCL == g_mathMode && xsize > 8 && ysize > 8)
        {
            std::vector<float> orignChannel;
            orignChannel.resize(xsize * ysize);
            memcpy(orignChannel.data(), channel, xsize * ysize * sizeof(float));
            _Blur(xsize, ysize, channel, sigma, border_ratio);
            tclBlur(orignChannel.data(), xsize, ysize, sigma, border_ratio, channel);
        }
        else
        {
            _Blur(xsize, ysize, channel, sigma, border_ratio);
        }
    }

    void OpsinDynamicsImage(size_t xsize, size_t ysize,
        std::vector<std::vector<float> > &rgb)
    {
        if (MODE_OPENCL == g_mathMode && xsize > 100 && ysize > 100)
        {
            float * r = rgb[0].data();
            float * g = rgb[1].data();
            float * b = rgb[2].data();

            clOpsinDynamicsImage(r, g, b, xsize, ysize);
        }
#ifdef __USE_CUDA__
        else if (MODE_CUDA == g_mathMode && xsize > 100 && ysize > 100)
        {
            float * r = rgb[0].data();
            float * g = rgb[1].data();
            float * b = rgb[2].data();

            cuOpsinDynamicsImage(r, g, b, xsize, ysize);
        }
#endif
        else if (MODE_CHECKCL == g_mathMode && xsize > 8 & ysize > 8)
        {
            std::vector< std::vector<float>> orig_rgb = rgb;
            _OpsinDynamicsImage(xsize, ysize, rgb);
            tclOpsinDynamicsImage(orig_rgb[0].data(), orig_rgb[1].data(), orig_rgb[2].data(), 
                    xsize, ysize,
                    rgb[0].data(), rgb[1].data(), rgb[2].data());
        }  
        else
        {
            _OpsinDynamicsImage(xsize, ysize, rgb);
        }
    }
}