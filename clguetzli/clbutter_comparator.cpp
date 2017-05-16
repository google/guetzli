#include "clbutter_comparator.h"
#include "clguetzli.h"
#include "clguetzli_test.h"

namespace butteraugli
{
    clButteraugliComparator::clButteraugliComparator(size_t xsize, size_t ysize, int step)
        : ButteraugliComparator(xsize, ysize, step)
    {

    }

    void clButteraugliComparator::DiffmapOpsinDynamicsImage(const std::vector<std::vector<float>> &xyb0,
        std::vector<std::vector<float>> &xyb1,
        std::vector<float> &result)
    {
        if (g_useOpenCL && xsize_ > 100 && ysize_ > 100)
        {
            result.resize(xsize_ * ysize_);
            clDiffmapOpsinDynamicsImage(xyb0[0].data(), xyb0[1].data(), xyb0[2].data(),
                xyb1[0].data(), xyb1[1].data(), xyb1[2].data(), xsize_, ysize_, step_, result.data());
        }
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

        if (g_checkOpenCL)
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

        if (g_checkOpenCL)
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
        std::vector<float> orign_ac;
        if (g_checkOpenCL)
        {
            orign_ac = *block_diff_ac;
        }

        ButteraugliComparator::EdgeDetectorLowFreq(xyb0, xyb1, block_diff_ac);

        if (g_checkOpenCL)
        {
            tclEdgeDetectorLowFreq(xyb0[0].data(), xyb0[1].data(), xyb0[2].data(),
                xyb1[0].data(), xyb1[1].data(), xyb1[2].data(),
                xsize_, ysize_, step_,
                orign_ac.data(), (*block_diff_ac).data());
        }
    }

    void clButteraugliComparator::CombineChannels(const std::vector<std::vector<float> >& mask_xyb,
        const std::vector<std::vector<float> >& mask_xyb_dc,
        const std::vector<float>& block_diff_dc,
        const std::vector<float>& block_diff_ac,
        const std::vector<float>& edge_detector_map,
        std::vector<float>* result)
    {
        std::vector<float> temp;
        if (g_checkOpenCL)
        {
            temp = *result;
        }

        ButteraugliComparator::CombineChannels(mask_xyb, mask_xyb_dc, block_diff_dc, block_diff_ac, edge_detector_map, result);

        if (g_checkOpenCL)
        {
			temp.resize(res_xsize_ * res_ysize_);
            tclCombineChannels(mask_xyb[0].data(), mask_xyb[1].data(), mask_xyb[2].data(),
                mask_xyb_dc[0].data(), mask_xyb_dc[1].data(), mask_xyb_dc[2].data(),
                block_diff_dc.data(),
                block_diff_ac.data(), edge_detector_map.data(), xsize_, ysize_, res_xsize_, res_ysize_, step_, &temp[0], &(*result)[0]);
        }
    }

    void MinSquareVal(size_t square_size, size_t offset, size_t xsize, size_t ysize, float *values) 
    {
        std::vector<float> img;
        if (g_checkOpenCL)
        {
            img.resize(xsize * ysize);
            memcpy(img.data(), values, xsize * ysize * sizeof(float));
        }

        _MinSquareVal(square_size, offset, xsize, ysize, values);


        if (g_checkOpenCL)
        {
            tclMinSquareVal(img.data(), square_size, offset, xsize, ysize, values);
        }
    }

    void Average5x5(int xsize, int ysize, std::vector<float>* diffs)
    {
        std::vector<float> diffs_org;
        if (g_checkOpenCL)
        {
            diffs_org = *diffs;
        }

        _Average5x5(xsize, ysize, diffs);

        if (g_checkOpenCL)
        {
            tclAverage5x5(xsize, ysize, diffs_org, *diffs);
        }
    }

    void DiffPrecompute(const std::vector<std::vector<float> > &xyb0, const std::vector<std::vector<float> > &xyb1, size_t xsize, size_t ysize, std::vector<std::vector<float> > *mask)
    {
        _DiffPrecompute(xyb0, xyb1, xsize, ysize, mask);

        if (g_checkOpenCL)
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
        if (g_useOpenCL)
        {
            mask->resize(3);
            mask_dc->resize(3);
            for (int i = 0; i < 3; i++)
            {
                (*mask)[i].resize(xsize * ysize);
                (*mask_dc)[i].resize(xsize * ysize);
            }
            clMask(xyb0[0].data(), xyb0[1].data(), xyb0[2].data(),
                xyb1[0].data(), xyb1[1].data(), xyb1[2].data(),
                xsize, ysize,
                (*mask)[0].data(), (*mask)[1].data(), (*mask)[2].data(),
                (*mask_dc)[0].data(), (*mask_dc)[1].data(), (*mask_dc)[2].data());
            return;
        }

        _Mask(xyb0, xyb1, xsize, ysize, mask, mask_dc);

        if (g_checkOpenCL)
        {
            tclMask(xyb0[0].data(), xyb0[1].data(), xyb0[2].data(),
                xyb1[0].data(), xyb1[1].data(), xyb1[2].data(),
                xsize, ysize,
                (*mask)[0].data(), (*mask)[1].data(), (*mask)[2].data(),
                (*mask_dc)[0].data(), (*mask_dc)[1].data(), (*mask_dc)[2].data());
        }
    }

    void CalculateDiffmap(const size_t xsize, const size_t ysize,
        const size_t step,
        std::vector<float>* diffmap)
    {
        std::vector<float> diffmap_org;
        if (g_checkOpenCL)
        {
            diffmap_org = *diffmap;
        }

        _CalculateDiffmap(xsize, ysize, step, diffmap);

        if (g_checkOpenCL)
        {
            tclCalculateDiffmap(xsize, ysize, step, diffmap_org.data(), diffmap_org.size(), (*diffmap).data());
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

        if (g_checkOpenCL)
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
        std::vector<float> result_org;
        if (g_checkOpenCL)
        {
            result_org = *result;
        }

        _ScaleImage(scale, result);

        if (g_checkOpenCL)
        {
            tclScaleImage(scale, result_org.data(), (*result).data(), (*result).size());
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

        if (g_checkOpenCL)
        {
            tclConvolution(xsize, ysize, xstep, len, offset, multipliers, inp, border_ratio, result);
        }
    }

    void Blur(size_t xsize, size_t ysize, float* channel, double sigma,
        double border_ratio)
    {
        std::vector<float> orignChannel;
        if (g_checkOpenCL)
        {
            orignChannel.resize(xsize * ysize);
            memcpy(orignChannel.data(), channel, xsize * ysize * sizeof(float));
        }

        _Blur(xsize, ysize, channel, sigma, border_ratio);

        if (g_checkOpenCL)
        {
            tclBlur(orignChannel.data(), xsize, ysize, sigma, border_ratio, channel);
        }
    }

    void OpsinDynamicsImage(size_t xsize, size_t ysize,
        std::vector<std::vector<float> > &rgb)
    {
        if (g_useOpenCL && xsize > 100 && ysize > 100)
        {
            float * r = rgb[0].data();
            float * g = rgb[1].data();
            float * b = rgb[2].data();

            clOpsinDynamicsImage(xsize, ysize, r, g, b);
        }
        else
        {
            std::vector< std::vector<float>> orig_rgb;
            if (g_checkOpenCL)
            {
                orig_rgb = rgb;
            }

            _OpsinDynamicsImage(xsize, ysize, rgb);

            if (g_checkOpenCL)
            {
                tclOpsinDynamicsImage(orig_rgb[0].data(), orig_rgb[1].data(), orig_rgb[2].data(), xsize, ysize,
                    rgb[0].data(), rgb[1].data(), rgb[2].data());
            }
        }  
    }
}