//#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#elif defined(cl_amd_fp64)
//#pragma OPENCL EXTENSION cl_amd_fp64 : enable
//#else
//#error "Double precision floating point not supported by OpenCL implementation."
//#endif

__kernel void MinSquareVal(__global float* pA, __global float* pC, int square_size, int offset)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int width = get_global_size(0);
	const int height = get_global_size(1);

	int minH = offset > y ? 0 : y - offset;
	int maxH = min(y + square_size - offset, height);

	int minW = offset > x ? 0 : x - offset;
	int maxW = min(x + square_size - offset, width);

	float minValue = pA[minH * width + minW];

	for (int j = minH; j < maxH; j++)
	{
		for (int i = minW; i < maxW; i++)
		{
			float tmp = pA[j * width + i];
			if (tmp < minValue) minValue = tmp;
		}
	}

	pC[y * width + x] = minValue;
}

__kernel void Convolution(__global float* multipliers, __global float* inp, __global float* result,
							int xsize, int xstep, int len, int offset, float border_ratio)
{
	const int ox = get_global_id(0);
	const int y = get_global_id(1);

	const int oxsize = get_global_size(0);
	const int ysize = get_global_size(1);

	const int x = ox * xstep;

	float weight_no_border = 0;
	for (int j = 0; j <= 2 * offset; j++)
	{
		weight_no_border += multipliers[j];
	}

	int minx = x < offset ? 0 : x - offset;
	int maxx = min(xsize, x + len - offset);

	float weight = 0.0;
	for (int j = minx; j < maxx; j++)
	{
		weight += multipliers[j - x + offset];
	}

	weight = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
	float scale = 1.0 / weight;

	float sum = 0.0;
	for (int j = minx; j < maxx; j++)
	{
		sum += inp[y * xsize + j] * multipliers[j - x + offset];
	}

	result[ox * ysize + y] = sum * scale;
}
/*
__kernel void ConvolutionX(__global float* multipliers, __global float* inp, __global float* result,
	int len, int offset, float border_ratio)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int xsize = get_global_size(0);
	const int ysize = get_global_size(1);

	float weight_no_border = 0;
	for (int j = 0; j <= 2 * offset; j++)
	{
		weight_no_border += multipliers[j];
	}

	int minx = x < offset ? 0 : x - offset;
	int maxx = min(xsize, x + len - offset);

	int miny = y < offset ? 0 : y - offset;
	int maxy = min(ysize, y + len - offset);

	float weightX = 0.0;
	for (int j = minx; j < maxx; j++)
	{
		weightX += multipliers[j - x + offset];
	}

	weightX = (1.0 - border_ratio) * weightX + border_ratio * weight_no_border;

	float weightY = 0.0;
	for (int j = miny; j < maxy; j++)
	{
		weightY += multipliers[j - y + offset];
	}

	weightY = (1.0 - border_ratio) * weightY + border_ratio * weight_no_border;


	float sum = 0.0;
	for (int j = miny; j < maxy; j++)
	{
		float sumx = 0.0;
		for (int i = minx; i < maxx; i++)
		{
			sumx += inp[j * xsize + i] * multipliers[i - x + offset];
		}

		sum += sumx * multipliers[j - y + offset];
	}

	result[y * xsize + x] = sum / weightY / weightX;
}
*/

__kernel void ConvolutionX(__global float* multipliers, __global float* inp, __global float* result,
	int len, int offset, float border_ratio)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int xsize = get_global_size(0);
	const int ysize = get_global_size(1);

	float weight_no_border = 0;
	for (int j = 0; j <= 2 * offset; j++)
	{
		weight_no_border += multipliers[j];
	}

	int minx = x < offset ? 0 : x - offset;
	int maxx = min(xsize, x + len - offset);

	float weight = 0.0;
	for (int j = minx; j < maxx; j++)
	{
		weight += multipliers[j - x + offset];
	}

	weight = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
	float scale = 1.0 / weight;

	float sum = 0.0;
	for (int j = minx; j < maxx; j++)
	{
		sum += inp[y * xsize + j] * multipliers[j - x + offset];
	}

	result[x * ysize + y] = sum * scale;
}

__kernel void ConvolutionY(__global float* multipliers, __global float* inp, __global float* result,
	int len, int offset, float border_ratio)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int xsize = get_global_size(0);
	const int ysize = get_global_size(1);

	float weight_no_border = 0;
	for (int j = 0; j <= 2 * offset; j++)
	{
		weight_no_border += multipliers[j];
	}

	int miny = y < offset ? 0 : y - offset;
	int maxy = min(ysize, y + len - offset);

	float weight = 0.0;
	for (int j = miny; j < maxy; j++)
	{
		weight += multipliers[j - y + offset];
	}

	weight = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
	float scale = 1.0 / weight;

	float sum = 0.0;
	for (int j = miny; j < maxy; j++)
	{
		sum += inp[j * xsize + x] * multipliers[j - y + offset];
	}

	result[y * xsize + x] = sum * scale;
}

__kernel void DownSample(__global float* pA, __global float* pC, int xstep, int ystep)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int xsize = get_global_size(0);
	const int ysize = get_global_size(1);

	const int oxsize = xsize / xstep;

	const int sample_x = x / xstep;
	const int sample_y = y / ystep;

	pC[y * xsize + x] = pA[sample_y * oxsize + sample_x];
}

void OpsinAbsorbance(const double in[3], double out[3])
{
	const float mix[12] = {
	0.348036746003,
	0.577814843137,
	0.0544556093735,
	0.774145581713,
	0.26922717275,
	0.767247733938,
	0.0366922708552,
	0.920130265014,
	0.0882062883536,
	0.158581714673,
	0.712857943858,
	10.6524069248,
	};

	out[0] = mix[0] * in[0] + mix[1] * in[1] + mix[2] * in[2] + mix[3];
	out[1] = mix[4] * in[0] + mix[5] * in[1] + mix[6] * in[2] + mix[7];
	out[2] = mix[8] * in[0] + mix[9] * in[1] + mix[10] * in[2] + mix[11];
}

double EvaluatePolynomial(const double x, const double *coefficients, int n)
{
	double b1 = 0.0;
	double b2 = 0.0;

	for (int i = n - 1; i >= 0; i--)
	{
		if (i == 0) {
			const double x_b1 = x * b1;
			b1 = x_b1 - b2 + coefficients[0];
			break;
		}
		const double x_b1 = x * b1;
		const double t = (x_b1 + x_b1) - b2 + coefficients[i];
		b2 = b1;
		b1 = t;
	}

	return b1;
}

double Gamma(double v)
{
	double min_value = 0.770000000000000;
	double max_value = 274.579999999999984;

	const double p[5 + 1] = {
		881.979476556478289, 1496.058452015812463, 908.662212739659481,
		373.566100223287378, 85.840860336314364, 6.683258861509244,
	};
	const double q[5 + 1] = {
		12.262350348616792, 20.557285797683576, 12.161463238367844,
		4.711532733641639, 0.899112889751053, 0.035662329617191,
	};

	const double x01 = (v - min_value) / (max_value - min_value);
	const double xc = 2.0 * x01 - 1.0;

	const double yp = EvaluatePolynomial(xc, p, 6);
	const double yq = EvaluatePolynomial(xc, q, 6);
	if (yq == 0.0) return 0.0;
	return (float)(yp / yq);
}

void RgbToXyb(double r, double g, double b, double *valx, double *valy, double *valz)
{
	const double a0 = 1.01611726948;
	const double a1 = 0.982482243696;
	const double a2 = 1.43571362627;
	const double a3 = 0.896039849412;
	*valx = a0 * r - a1 * g;
	*valy = a2 * r + a3 * g;
	*valz = b;
}

__kernel void OpsinDynamicsImage(
	__global float *r, __global float *g, __global float *b,
	__global float *r_blurred, __global float *g_blurred, __global float *b_blurred,
	int size)
{
	const int i = get_global_id(0);
	double pre[3] = { r_blurred[i], g_blurred[i],  b_blurred[i] };
	double pre_mixed[3];
	OpsinAbsorbance(pre, pre_mixed);
	double sensitivity[3];
	sensitivity[0] = Gamma(pre_mixed[0]) / pre_mixed[0];
	sensitivity[1] = Gamma(pre_mixed[1]) / pre_mixed[1];
	sensitivity[2] = Gamma(pre_mixed[2]) / pre_mixed[2];

	double cur_rgb[3] = { r_blurred[i], g_blurred[i],  b_blurred[i] };
	double cur_mixed[3];
    OpsinAbsorbance(cur_rgb, cur_mixed);
    cur_mixed[0] *= sensitivity[0];
    cur_mixed[1] *= sensitivity[1];
    cur_mixed[2] *= sensitivity[2];
    double x, y, z;
	RgbToXyb(cur_mixed[0], cur_mixed[1], cur_mixed[2], &x, &y, &z);
    r[i] = x;
    g[i] = y;
    b[i] = z;
}


double InterpolateClampNegative(const double *array,
	int size, double sx) {
	if (sx < 0) {
		sx = 0;
	}
	double ix = fabs(sx);
	int baseix = (int)(ix);
	double res;
	if (baseix >= size - 1) {
		res = array[size - 1];
	}
	else {
		double mix = ix - baseix;
		int nextix = baseix + 1;
		res = array[baseix] + mix * (array[nextix] - array[baseix]);
	}
	return res;
}

void MakeMask(double extmul, double extoff,
	double mul, double offset,
	double scaler, double *result)
{
	for (size_t i = 0; i < 512; ++i) {
		const double c = mul / ((0.01 * scaler * i) + offset);
		result[i] = 1.0 + extmul * (c + extoff);
		result[i] *= result[i];
	}
}

double MaskX(double delta) {
	const double extmul = 0.975741017749;
	const double extoff = -4.25328244168;
	const double offset = 0.454909521427;
	const double scaler = 0.0738288224836;
	const double mul = 20.8029176447;
	double lut[512];
	MakeMask(extmul, extoff, mul, offset, scaler, lut);
	return InterpolateClampNegative(lut, 512, delta);
}

double MaskY(double delta) {
	const double extmul = 0.373995618954;
	const double extoff = 1.5307267433;
	const double offset = 0.911952641929;
	const double scaler = 1.1731667845;
	const double mul = 16.2447033988;
	double lut[512];
	MakeMask(extmul, extoff, mul, offset, scaler, lut);
	return InterpolateClampNegative(lut, 512, delta);
}

double MaskB(double delta) {
	const double extmul = 0.61582234137;
	const double extoff = -4.25376118646;
	const double offset = 1.05105070921;
	const double scaler = 0.47434643535;
	const double mul = 31.1444967089;
	double lut[512];
	MakeMask(extmul, extoff, mul, offset, scaler, lut);
	return InterpolateClampNegative(lut, 512, delta);
}

double MaskDcX(double delta) {
	const double extmul = 1.79116943438;
	const double extoff = -3.86797479189;
	const double offset = 0.670960225853;
	const double scaler = 0.486575865525;
	const double mul = 20.4563479139;
	double lut[512];
	MakeMask(extmul, extoff, mul, offset, scaler, lut);
	return InterpolateClampNegative(lut, 512, delta);
}

double MaskDcY(double delta) {
	const double extmul = 0.212223514236;
	const double extoff = -3.65647120524;
	const double offset = 1.73396799447;
	const double scaler = 0.170392660501;
	const double mul = 21.6566724788;
	double lut[512];
	MakeMask(extmul, extoff, mul, offset, scaler, lut);
	return InterpolateClampNegative(lut, 512, delta);
}

double MaskDcB(double delta) {
	const double extmul = 0.349376011816;
	const double extoff = -0.894711072781;
	const double offset = 0.901647926679;
	const double scaler = 0.380086095024;
	const double mul = 18.0373825149;
	double lut[512];
	MakeMask(extmul, extoff, mul, offset, scaler, lut);
	return InterpolateClampNegative(lut, 512, delta);
}

__kernel void DoMask(
	__global float *mask_x, __global float *mask_y, __global float *mask_b,
	__global float *mask_dc_x, __global float *mask_dc_y, __global float *mask_dc_b,
	int xsize, int ysize)
{
	const double w00 = 232.206464018;
	const double w11 = 22.9455222245;
	const double w22 = 503.962310606;

	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const size_t idx = y * xsize + x;
	const double s0 = mask_x[idx];
	const double s1 = mask_y[idx];
	const double s2 = mask_b[idx];
	const double p0 = w00 * s0;
	const double p1 = w11 * s1;
	const double p2 = w22 * s2;

	mask_x[idx] = (float)(MaskX(p0));
	mask_y[idx] = (float)(MaskY(p1));
	mask_b[idx] = (float)(MaskB(p2));
	mask_dc_x[idx] = (float)(MaskDcX(p0));
	mask_dc_y[idx] = (float)(MaskDcY(p1));
	mask_dc_b[idx] = (float)(MaskDcB(p2));

}

__kernel void ScaleImage(float scale, __global float *result)
{
	const int i = get_global_id(0);
	result[i] *= scale;
}

double DotProduct(float u[3], double v[3]) {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

__kernel void CombineChannels(
	__global float *mask_x, __global float *mask_y, __global float *mask_b,
	__global float *mask_dc_x, __global float *mask_dc_y, __global float *mask_dc_b,
	__global float *block_diff_dc,
	__global float *block_diff_ac,
	__global float *edge_detector_map,
	int xsize, int ysize,
	int step,
	__global float *result)
{
	const int res_x = get_global_id(0);
	const int res_y = get_global_id(1);

	const int res_xsize = get_global_size(0);
	const int res_ysize = get_global_size(1);

	if (res_x * step >= xsize - (8 - step)) return;
	if (res_y * step >= ysize - (8 - step)) return;

	double mask[3];
	double dc_mask[3];
	mask[0] = mask_x[(res_y + 3) * xsize + (res_x + 3)];
	dc_mask[0] = mask_dc_x[(res_y + 3) * xsize + (res_x + 3)];

	mask[1] = mask_y[(res_y + 3) * xsize + (res_x + 3)];
	dc_mask[1] = mask_dc_y[(res_y + 3) * xsize + (res_x + 3)];

	mask[1] = mask_b[(res_y + 3) * xsize + (res_x + 3)];
	dc_mask[1] = mask_dc_b[(res_y + 3) * xsize + (res_x + 3)];

	size_t res_ix = (res_y * res_xsize + res_x) / step;
	result[res_ix] = (float)(
		DotProduct((float *)&block_diff_dc[3 * res_ix], dc_mask) +
		DotProduct((float *)&block_diff_ac[3 * res_ix], mask) +
		DotProduct((float *)&edge_detector_map[3 * res_ix], mask));
}

inline double Interpolate(__constant double *array, int size, double sx) {
	double ix = fabs(sx);

	int baseix = (int)(ix);
	double res;
	if (baseix >= size - 1) {
		res = array[size - 1];
	}
	else {
		double mix = ix - baseix;
		int nextix = baseix + 1;
		res = array[baseix] + mix * (array[nextix] - array[baseix]);
	}
	if (sx < 0) res = -res;
	return res;
}

__constant double XybLowFreqToVals_inc = 5.2511644570349185;
__constant double XybLowFreqToVals_lut[21] = {
	0,
	1 * XybLowFreqToVals_inc,
	2 * XybLowFreqToVals_inc,
	3 * XybLowFreqToVals_inc,
	4 * XybLowFreqToVals_inc,
	5 * XybLowFreqToVals_inc,
	6 * XybLowFreqToVals_inc,
	7 * XybLowFreqToVals_inc,
	8 * XybLowFreqToVals_inc,
	9 * XybLowFreqToVals_inc,
	10 * XybLowFreqToVals_inc,
	11 * XybLowFreqToVals_inc,
	12 * XybLowFreqToVals_inc,
	13 * XybLowFreqToVals_inc,
	14 * XybLowFreqToVals_inc,
	15 * XybLowFreqToVals_inc,
	16 * XybLowFreqToVals_inc,
	17 * XybLowFreqToVals_inc,
	18 * XybLowFreqToVals_inc,
	19 * XybLowFreqToVals_inc,
	20 * XybLowFreqToVals_inc,
};

void XybLowFreqToVals(double x, double y, double z,
	double *valx, double *valy, double *valz) {
	const double xmul = 6.64482198135;
	const double ymul = 0.837846224276;
	const double zmul = 7.34905756986;
	const double y_to_z_mul = 0.0812519812628;
	z += y_to_z_mul * y;
	*valz = z * zmul;
	*valx = x * xmul;
	*valy = Interpolate(&XybLowFreqToVals_lut[0], 21, y * ymul);
}

void XybDiffLowFreqSquaredAccumulate(double r0, double g0, double b0,
	double r1, double g1, double b1,
	double factor, double res[3]) {
	double valx0, valy0, valz0;
	double valx1, valy1, valz1;
	XybLowFreqToVals(r0, g0, b0, &valx0, &valy0, &valz0);
	if (r1 == 0.0 && g1 == 0.0 && b1 == 0.0) {
		//PROFILER_ZONE("XybDiff r1=g1=b1=0");
		res[0] += factor * valx0 * valx0;
		res[1] += factor * valy0 * valy0;
		res[2] += factor * valz0 * valz0;
		return;
	}
	XybLowFreqToVals(r1, g1, b1, &valx1, &valy1, &valz1);
	// Approximate the distance of the colors by their respective distances
	// to gray.
	double valx = valx0 - valx1;
	double valy = valy0 - valy1;
	double valz = valz0 - valz1;
	res[0] += factor * valx * valx;
	res[1] += factor * valy * valy;
	res[2] += factor * valz * valz;
}

__kernel void edgeDetectorMap(__global float *result,
						      __global float *r, __global float *g, __global float* b,
						      __global float *r2, __global float* g2, __global float *b2,
						     int xsize, int ysize, int step)
{
	const int res_x = get_global_id(0);
	const int res_y = get_global_id(1);

	const int res_xsize = get_global_size(0);
	const int res_ysize = get_global_size(1);

	int pos_x = res_x * step;
	int pos_y = res_y * step;

	if (pos_x >= xsize - (8 - step)) return;
	if (pos_y >= ysize - (8 - step)) return;

	pos_x = min(pos_x, xsize - 8);
	pos_y = min(pos_y, ysize - 8);

	int local_count = 0;
	double local_xyb[3] = { 0 };
	const double w = 0.711100840192;

	int offset[4][2] = {{0,0}, {0,7}, {7,0}, {7,7}};
	int edgeSize = 3;

	for (int k = 0; k < 4; k++)
	{
		int x = pos_x + offset[k][0];
		int y = pos_y + offset[k][1];

		if (x >= edgeSize && x + edgeSize < xsize) {
			size_t ix = y * xsize + (x - edgeSize);
			size_t ix2 = ix + 2 * edgeSize;
			XybDiffLowFreqSquaredAccumulate(
				w * (r[ix] - r[ix2]),
				w * (g[ix] - g[ix2]),
				w * (b[ix] - b[ix2]),
				w * (r2[ix] - r2[ix2]),
				w * (g2[ix] - g2[ix2]),
				w * (b2[ix] - b2[ix2]),
				1.0, local_xyb);
			++local_count;
		}
		if (y >= edgeSize && y + edgeSize < ysize) {
			size_t ix = (y - edgeSize) * xsize + x;
			size_t ix2 = ix + 2 * edgeSize * xsize;
			XybDiffLowFreqSquaredAccumulate(
				w * (r[ix] - r[ix2]),
				w * (g[ix] - g[ix2]),
				w * (b[ix] - b[ix2]),
				w * (r2[ix] - r2[ix2]),
				w * (g2[ix] - g2[ix2]),
				w * (b2[ix] - b2[ix2]),
				1.0, local_xyb);
			++local_count;
		}
	}

	const double weight = 0.01617112696;
	const double mul = weight * 8.0 / local_count;

	int idx = (res_y * res_xsize + res_x) * 3;
	result[idx]     = local_xyb[0];
	result[idx + 1] = local_xyb[1];
	result[idx + 2] = local_xyb[2];
}

__kernel void edgeDetectorLowFreq(__global float *result,
	__global float *r, __global float *g, __global float* b,
	__global float *r2, __global float* g2, __global float *b2,
	int xsize, int ysize, int step)
{
	const int res_x = get_global_id(0);
	const int res_y = get_global_id(1);

	if (res_x < 8 / step) return;

	const int res_xsize = get_global_size(0);
	const int res_ysize = get_global_size(1);

	int pos_x = (res_x - (8 / step)) * step;
	int pos_y = res_y * step;

	if (pos_x + 8 >= xsize) return;
	if (pos_y + 8 >= ysize) return;

	int ix = pos_y * xsize + pos_x;

	double diff[4][3];
	__global float* blurred0[3] = { r, g, b };
	__global float* blurred1[3] = { r2, g2, b2 };

	for (int i = 0; i < 3; ++i) {
		int ix2 = ix + 8;
		diff[0][i] =
			((blurred1[i][ix] - blurred0[i][ix]) +
			(blurred0[i][ix2] - blurred1[i][ix2]));
		ix2 = ix + 8 * xsize;
		diff[1][i] =
			((blurred1[i][ix] - blurred0[i][ix]) +
			(blurred0[i][ix2] - blurred1[i][ix2]));
		ix2 = ix + 6 * xsize + 6;
		diff[2][i] =
			((blurred1[i][ix] - blurred0[i][ix]) +
			(blurred0[i][ix2] - blurred1[i][ix2]));
		ix2 = ix + 6 * xsize - 6;
		diff[3][i] = pos_x < 8 ? 0 :
			((blurred1[i][ix] - blurred0[i][ix]) +
			(blurred0[i][ix2] - blurred1[i][ix2]));
	}
	double max_diff_xyb[3] = { 0 };
	for (int k = 0; k < 4; ++k) {
		double diff_xyb[3] = { 0 };
		XybDiffLowFreqSquaredAccumulate(diff[k][0], diff[k][1], diff[k][2],
			0, 0, 0, 1.0,
			diff_xyb);
		for (int i = 0; i < 3; ++i) {
			max_diff_xyb[i] = max(max_diff_xyb[i], diff_xyb[i]);
		}
	}

	int res_ix = res_y * res_xsize + res_x;

	const double kMul = 10;

	result[res_ix * 3] = max_diff_xyb[0] * kMul;
	result[res_ix * 3 + 1] = max_diff_xyb[1] * kMul;
	result[res_ix * 3 + 2] = max_diff_xyb[2] * kMul;
}

#define kBlockEdge 8
#define kBlockSize (kBlockEdge * kBlockEdge)
#define kBlockEdgeHalf  (kBlockEdge / 2)
#define kBlockHalf (kBlockEdge * kBlockEdgeHalf)

__constant double csf8x8[kBlockHalf + kBlockEdgeHalf + 1] = {
	5.28270670524,
	0.0,
	0.0,
	0.0,
	0.3831134973,
	0.676303603859,
	3.58927792424,
	18.6104367002,
	18.6104367002,
	3.09093131948,
	1.0,
	0.498250875965,
	0.36198671102,
	0.308982169883,
	0.1312701920435,
	2.37370549629,
	3.58927792424,
	1.0,
	2.37370549629,
	0.991205724152,
	1.05178802919,
	0.627264168628,
	0.4,
	0.1312701920435,
	0.676303603859,
	0.498250875965,
	0.991205724152,
	0.5,
	0.3831134973,
	0.349686450518,
	0.627264168628,
	0.308982169883,
	0.3831134973,
	0.36198671102,
	1.05178802919,
	0.3831134973,
	0.12,
};

typedef struct __Complex
{
	double real;
	double imag;
}Complex;

constant double kSqrtHalf = 0.70710678118654752440084436210484903;

void RealFFT8(const double* in, Complex* out) {
	double t1, t2, t3, t5, t6, t7, t8;
	t8 = in[6];
	t5 = in[2] - t8;
	t8 += in[2];
	out[2].real = t8;
	out[6].imag = -t5;
	out[4].imag = t5;
	t8 = in[4];
	t3 = in[0] - t8;
	t8 += in[0];
	out[0].real = t8;
	out[4].real = t3;
	out[6].real = t3;
	t7 = in[5];
	t3 = in[1] - t7;
	t7 += in[1];
	out[1].real = t7;
	t8 = in[7];
	t5 = in[3] - t8;
	t8 += in[3];
	out[3].real = t8;
	t2 = -t5;
	t6 = t3 - t5;
	t8 = kSqrtHalf;
	t6 *= t8;
	out[5].real = out[4].real - t6;
	t1 = t3 + t5;
	t1 *= t8;
	out[5].imag = out[4].imag - t1;
	t6 += out[4].real;
	out[4].real = t6;
	t1 += out[4].imag;
	out[4].imag = t1;
	t5 = t2 - t3;
	t5 *= t8;
	out[7].imag = out[6].imag - t5;
	t2 += t3;
	t2 *= t8;
	out[7].real = out[6].real - t2;
	t2 += out[6].real;
	out[6].real = t2;
	t5 += out[6].imag;
	out[6].imag = t5;
	t5 = out[2].real;
	t1 = out[0].real - t5;
	t7 = out[3].real;
	t5 += out[0].real;
	t3 = out[1].real - t7;
	t7 += out[1].real;
	t8 = t5 + t7;
	out[0].real = t8;
	t5 -= t7;
	out[1].real = t5;
	out[2].imag = t3;
	out[3].imag = -t3;
	out[3].real = t1;
	out[2].real = t1;
	out[0].imag = 0;
	out[1].imag = 0;

	// Reorder to the correct output order.
	// TODO: Modify the above computation so that this is not needed.
	Complex tmp = out[2];
	out[2] = out[3];
	out[3] = out[5];
	out[5] = out[7];
	out[7] = out[4];
	out[4] = out[1];
	out[1] = out[6];
	out[6] = tmp;
}

void TransposeBlock(Complex data[kBlockSize]) {
	for (int i = 0; i < kBlockEdge; i++) {
		for (int j = 0; j < i; j++) {
			Complex tmp = data[kBlockEdge * i + j];
			data[kBlockEdge * i + j] = data[kBlockEdge * j + i];
			data[kBlockEdge * j + i] = tmp;
		}
	}
}

//  D. J. Bernstein's Fast Fourier Transform algorithm on 4 elements.
inline void FFT4(Complex* a) {
	double t1, t2, t3, t4, t5, t6, t7, t8;
	t5 = a[2].real;
	t1 = a[0].real - t5;
	t7 = a[3].real;
	t5 += a[0].real;
	t3 = a[1].real - t7;
	t7 += a[1].real;
	t8 = t5 + t7;
	a[0].real = t8;
	t5 -= t7;
	a[1].real = t5;
	t6 = a[2].imag;
	t2 = a[0].imag - t6;
	t6 += a[0].imag;
	t5 = a[3].imag;
	a[2].imag = t2 + t3;
	t2 -= t3;
	a[3].imag = t2;
	t4 = a[1].imag - t5;
	a[3].real = t1 + t4;
	t1 -= t4;
	a[2].real = t1;
	t5 += a[1].imag;
	a[0].imag = t6 + t5;
	t6 -= t5;
	a[1].imag = t6;
}

//  D. J. Bernstein's Fast Fourier Transform algorithm on 8 elements.
void FFT8(Complex* a) {
	double t1, t2, t3, t4, t5, t6, t7, t8;

	t7 = a[4].imag;
	t4 = a[0].imag - t7;
	t7 += a[0].imag;
	a[0].imag = t7;

	t8 = a[6].real;
	t5 = a[2].real - t8;
	t8 += a[2].real;
	a[2].real = t8;

	t7 = a[6].imag;
	a[6].imag = t4 - t5;
	t4 += t5;
	a[4].imag = t4;

	t6 = a[2].imag - t7;
	t7 += a[2].imag;
	a[2].imag = t7;

	t8 = a[4].real;
	t3 = a[0].real - t8;
	t8 += a[0].real;
	a[0].real = t8;

	a[4].real = t3 - t6;
	t3 += t6;
	a[6].real = t3;

	t7 = a[5].real;
	t3 = a[1].real - t7;
	t7 += a[1].real;
	a[1].real = t7;

	t8 = a[7].imag;
	t6 = a[3].imag - t8;
	t8 += a[3].imag;
	a[3].imag = t8;
	t1 = t3 - t6;
	t3 += t6;

	t7 = a[5].imag;
	t4 = a[1].imag - t7;
	t7 += a[1].imag;
	a[1].imag = t7;

	t8 = a[7].real;
	t5 = a[3].real - t8;
	t8 += a[3].real;
	a[3].real = t8;

	t2 = t4 - t5;
	t4 += t5;

	t6 = t1 - t4;
	t8 = kSqrtHalf;
	t6 *= t8;
	a[5].real = a[4].real - t6;
	t1 += t4;
	t1 *= t8;
	a[5].imag = a[4].imag - t1;
	t6 += a[4].real;
	a[4].real = t6;
	t1 += a[4].imag;
	a[4].imag = t1;

	t5 = t2 - t3;
	t5 *= t8;
	a[7].imag = a[6].imag - t5;
	t2 += t3;
	t2 *= t8;
	a[7].real = a[6].real - t2;
	t2 += a[6].real;
	a[6].real = t2;
	t5 += a[6].imag;
	a[6].imag = t5;

	FFT4(a);

	// Reorder to the correct output order.
	// TODO: Modify the above computation so that this is not needed.
	Complex tmp = a[2];
	a[2] = a[3];
	a[3] = a[5];
	a[5] = a[7];
	a[7] = a[4];
	a[4] = a[1];
	a[1] = a[6];
	a[6] = tmp;
}

double abssq(const Complex c) {
	return c.real * c.real + c.imag * c.imag;
}

void ButteraugliFFTSquared(double block[kBlockSize]) {
	double global_mul = 0.000064;
	Complex block_c[kBlockSize];

	for (int y = 0; y < kBlockEdge; ++y) {
		RealFFT8(block + y * kBlockEdge, block_c + y * kBlockEdge);
	}
	TransposeBlock(block_c);
	double r0[kBlockEdge];
	double r1[kBlockEdge];
	for (int x = 0; x < kBlockEdge; ++x) {
		r0[x] = block_c[x].real;
		r1[x] = block_c[kBlockHalf + x].real;
	}
	RealFFT8(r0, block_c);
	RealFFT8(r1, block_c + kBlockHalf);
	for (int y = 1; y < kBlockEdgeHalf; ++y) {
		FFT8(block_c + y * kBlockEdge);
	}
	for (int i = kBlockEdgeHalf; i < kBlockHalf + kBlockEdgeHalf + 1; ++i) {
		block[i] = abssq(block_c[i]);
		block[i] *= global_mul;
	}
}

__constant double MakeHighFreqColorDiffDy_off = 1.4103373714040413;
__constant double MakeHighFreqColorDiffDy_inc = 0.7084088867024;
__constant double MakeHighFreqColorDiffDy_lut[21] ={
	0.0,
	MakeHighFreqColorDiffDy_off,
	MakeHighFreqColorDiffDy_off + 1 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 2 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 3 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 4 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 5 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 6 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 7 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 8 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 9 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 10 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 11 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 12 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 13 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 14 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 15 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 16 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 17 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 18 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 19 * MakeHighFreqColorDiffDy_inc,
};

double RemoveRangeAroundZero(double v, double range) {
	if (v >= -range && v < range) {
		return 0;
	}
	if (v < 0) {
		return v + range;
	}
	else {
		return v - range;
	}
}

// Computes 8x8 FFT of each channel of xyb0 and xyb1 and adds the total squared
// 3-dimensional xybdiff of the two blocks to diff_xyb_{dc,ac} and the average
// diff on the edges to diff_xyb_edge_dc.
void ButteraugliBlockDiff(double xyb0[3 * kBlockSize],
	double xyb1[3 * kBlockSize],
	double diff_xyb_dc[3],
	double diff_xyb_ac[3],
	double diff_xyb_edge_dc[3]) {

	double avgdiff_xyb[3] = { 0.0 };
	double avgdiff_edge[3][4] = { { 0.0 } };
	for (int i = 0; i < 3 * kBlockSize; ++i) {
		const double diff_xyb = xyb0[i] - xyb1[i];
		const int c = i / kBlockSize;
		avgdiff_xyb[c] += diff_xyb / kBlockSize;
		const int k = i % kBlockSize;
		const int kx = k % kBlockEdge;
		const int ky = k / kBlockEdge;
		const int h_edge_idx = ky == 0 ? 1 : ky == 7 ? 3 : -1;
		const int v_edge_idx = kx == 0 ? 0 : kx == 7 ? 2 : -1;
		if (h_edge_idx >= 0) {
			avgdiff_edge[c][h_edge_idx] += diff_xyb / kBlockEdge;
		}
		if (v_edge_idx >= 0) {
			avgdiff_edge[c][v_edge_idx] += diff_xyb / kBlockEdge;
		}
	}
	XybDiffLowFreqSquaredAccumulate(avgdiff_xyb[0],
		avgdiff_xyb[1],
		avgdiff_xyb[2],
		0, 0, 0, csf8x8[0],
		diff_xyb_dc);
	for (int i = 0; i < 4; ++i) {
		XybDiffLowFreqSquaredAccumulate(avgdiff_edge[0][i],
			avgdiff_edge[1][i],
			avgdiff_edge[2][i],
			0, 0, 0, csf8x8[0],
			diff_xyb_edge_dc);
	}

	double* xyb_avg = xyb0;
	double* xyb_halfdiff = xyb1;
	for (int i = 0; i < 3 * kBlockSize; ++i) {
		double avg = (xyb0[i] + xyb1[i]) / 2;
		double halfdiff = (xyb0[i] - xyb1[i]) / 2;
		xyb_avg[i] = avg;
		xyb_halfdiff[i] = halfdiff;
	}
	double *y_avg = &xyb_avg[kBlockSize];
	double *x_halfdiff_squared = &xyb_halfdiff[0];
	double *y_halfdiff = &xyb_halfdiff[kBlockSize];
	double *z_halfdiff_squared = &xyb_halfdiff[2 * kBlockSize];
	ButteraugliFFTSquared(y_avg);
	ButteraugliFFTSquared(x_halfdiff_squared);
	ButteraugliFFTSquared(y_halfdiff);
	ButteraugliFFTSquared(z_halfdiff_squared);

	const double xmul = 64.8;
	const double ymul = 1.753123908348329;
	const double ymul2 = 1.51983458269;
	const double zmul = 2.4;

	for (size_t i = kBlockEdgeHalf; i < kBlockHalf + kBlockEdgeHalf + 1; ++i) {
		double d = csf8x8[i];
		diff_xyb_ac[0] += d * xmul * x_halfdiff_squared[i];
		diff_xyb_ac[2] += d * zmul * z_halfdiff_squared[i];

		y_avg[i] = sqrt(y_avg[i]);
		y_halfdiff[i] = sqrt(y_halfdiff[i]);
		double y0 = y_avg[i] - y_halfdiff[i];
		double y1 = y_avg[i] + y_halfdiff[i];
		// Remove the impact of small absolute values.
		// This improves the behavior with flat noise.
		const double ylimit = 0.04;
		y0 = RemoveRangeAroundZero(y0, ylimit);
		y1 = RemoveRangeAroundZero(y1, ylimit);
		if (y0 != y1) {
			double valy0 = Interpolate(&MakeHighFreqColorDiffDy_lut[0], 21, y0 * ymul2);
			double valy1 = Interpolate(&MakeHighFreqColorDiffDy_lut[0], 21, y1 * ymul2);
			double valy = ymul * (valy0 - valy1);
			diff_xyb_ac[1] += d * valy * valy;
		}
	}
}

__kernel void blockDiffMap(__global float* r, __global float* g, __global float* b,
	__global float* r2, __global float* g2, __global float* b2,
	__global float* block_diff_dc, __global float* block_diff_ac,
	int xsize, int ysize, int step)
{
	const int res_x = get_global_id(0);
	const int res_y = get_global_id(1);

	const int res_xsize = get_global_size(0);
	const int res_ysize = get_global_size(1);

	int pos_x = res_x * step;
	int pos_y = res_y * step;

	if ((pos_x + kBlockEdge - step - 1) >= ysize) return;
	if ((pos_y + kBlockEdge - step - 1) >= xsize) return;

	size_t res_ix = res_y * res_xsize + res_x;
	size_t offset = min(res_y * step, ysize - 8) * xsize + min(res_x * step, xsize - 8);

	double block0[3 * kBlockEdge * kBlockEdge];
	double block1[3 * kBlockEdge * kBlockEdge];

	double *block0_r = &block0[0];
	double *block0_g = &block0[kBlockEdge * kBlockEdge];
	double *block0_b = &block0[2 * kBlockEdge * kBlockEdge];

	double *block1_r = &block1[0];
	double *block1_g = &block1[kBlockEdge * kBlockEdge];
	double *block1_b = &block1[2 * kBlockEdge * kBlockEdge];

	for (int y = 0; y < kBlockEdge; y++)
	{
		for (int x = 0; x < kBlockEdge; x++)
		{
			block0_r[kBlockEdge * y + x] = r[offset + y * xsize + x];
			block0_g[kBlockEdge * y + x] = g[offset + y * xsize + x];
			block0_b[kBlockEdge * y + x] = b[offset + y * xsize + x];
			block1_r[kBlockEdge * y + x] = r2[offset + y * xsize + x];
			block1_g[kBlockEdge * y + x] = g2[offset + y * xsize + x];
			block1_b[kBlockEdge * y + x] = b2[offset + y * xsize + x];
		}
	}

	double diff_xyb_dc[3] = { 0.0 };
	double diff_xyb_ac[3] = { 0.0 };
	double diff_xyb_edge_dc[3] = { 0.0 };

	ButteraugliBlockDiff(block0, block1, diff_xyb_dc, diff_xyb_ac, diff_xyb_edge_dc);

	for (int i = 0; i < 3; i++)
	{
		block_diff_dc[3 * res_ix + i] = diff_xyb_dc[i];
		block_diff_ac[3 * res_ix + i] = diff_xyb_ac[i];
	}
}
__kernel void MaskHighIntensityChange(
	__global float *xyb0_x, __global float *xyb0_y, __global float *xyb0_b,
	__global float *xyb1_x, __global float *xyb1_y, __global float *xyb1_b,
	__global float *c0_x, __global float *c0_y, __global float *c0_b,
	__global float *c1_x, __global float *c1_y, __global float *c1_b
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int xsize = get_global_size(0);
	const int ysize = get_global_size(1);

	size_t ix = y * xsize + x;
	const double ave[3] = {
	(c0_x[ix] + c1_x[ix]) * 0.5,
	(c0_y[ix] + c1_y[ix]) * 0.5,
	(c0_b[ix] + c1_b[ix]) * 0.5,
	};
	double sqr_max_diff = -1;
	{
		int offset[4] =
			{ -1, 1, -(int)(xsize), (int)(xsize) };
		int border[4] =
			{ x == 0, x + 1 == xsize, y == 0, y + 1 == ysize };
		for (int dir = 0; dir < 4; ++dir) {
			if (border[dir]) {
			continue;
			}
			const int ix2 = ix + offset[dir];
			double diff = 0.5 * (c0_y[ix2] + c1_y[ix2]) - ave[1];
			diff *= diff;
			if (sqr_max_diff < diff) {
			sqr_max_diff = diff;
			}
		}
	}
	const double kReductionX = 275.19165240059317;
	const double kReductionY = 18599.41286306991;
	const double kReductionZ = 410.8995306951065;
	const double kChromaBalance = 106.95800948271017;
	double chroma_scale = kChromaBalance / (ave[1] + kChromaBalance);

	const double mix[3] = {
		chroma_scale * kReductionX / (sqr_max_diff + kReductionX),
		kReductionY / (sqr_max_diff + kReductionY),
		chroma_scale * kReductionZ / (sqr_max_diff + kReductionZ),
	};
	// Interpolate lineraly between the average color and the actual
	// color -- to reduce the importance of this pixel.
	xyb0_x[ix] = (float)(mix[0] * c0_x[ix] + (1 - mix[0]) * ave[0]);
	xyb1_x[ix] = (float)(mix[0] * c1_x[ix] + (1 - mix[0]) * ave[0]);

	xyb0_y[ix] = (float)(mix[1] * c0_y[ix] + (1 - mix[1]) * ave[1]);
	xyb1_y[ix] = (float)(mix[1] * c1_y[ix] + (1 - mix[1]) * ave[1]);

	xyb0_b[ix] = (float)(mix[2] * c0_b[ix] + (1 - mix[2]) * ave[2]);
	xyb1_b[ix] = (float)(mix[2] * c1_b[ix] + (1 - mix[2]) * ave[2]);
}


__constant double XybToVals_off = 11.38708334481672;
__constant double XybToVals_inc = 14.550189611520716;
__constant double XybToVals_lut[21] = {
	0,
	XybToVals_off,
	XybToVals_off + 1 * XybToVals_inc,
	XybToVals_off + 2 * XybToVals_inc,
	XybToVals_off + 3 * XybToVals_inc,
	XybToVals_off + 4 * XybToVals_inc,
	XybToVals_off + 5 * XybToVals_inc,
	XybToVals_off + 6 * XybToVals_inc,
	XybToVals_off + 7 * XybToVals_inc,
	XybToVals_off + 8 * XybToVals_inc,
	XybToVals_off + 9 * XybToVals_inc,
	XybToVals_off + 10 * XybToVals_inc,
	XybToVals_off + 11 * XybToVals_inc,
	XybToVals_off + 12 * XybToVals_inc,
	XybToVals_off + 13 * XybToVals_inc,
	XybToVals_off + 14 * XybToVals_inc,
	XybToVals_off + 15 * XybToVals_inc,
	XybToVals_off + 16 * XybToVals_inc,
	XybToVals_off + 17 * XybToVals_inc,
	XybToVals_off + 18 * XybToVals_inc,
	XybToVals_off + 19 * XybToVals_inc,
};

void XybToVals(
	double x, double y, double z,
	double *valx, double *valy, double *valz)
{
	const double xmul = 0.758304045695;
    const double ymul = 2.28148649801;
	const double zmul = 1.87816926918;

	*valx = Interpolate(&XybToVals_lut[0], 21, x * xmul);
	*valy = Interpolate(&XybToVals_lut[0], 21, y * ymul);
	*valz = zmul * z;
}

__kernel void DiffPrecompute(
	__global float *xyb0_x, __global float *xyb0_y, __global float *xyb0_b,
	__global float *xyb1_x, __global float *xyb1_y, __global float *xyb1_b,
	__global float *mask_x, __global float *mask_y, __global float *mask_b )
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int xsize = get_global_size(0);
	const int ysize = get_global_size(1);

	double valsh0[3] = { 0.0 };
	double valsv0[3] = { 0.0 };
	double valsh1[3] = { 0.0 };
	double valsv1[3] = { 0.0 };
	int ix2;

	size_t ix = x + xsize * y;
	if (x + 1 < xsize) {
		ix2 = ix + 1;
	}
	else {
		ix2 = ix - 1;
	}
	{
		double x0 = (xyb0_x[ix] - xyb0_x[ix2]);
		double y0 = (xyb0_y[ix] - xyb0_y[ix2]);
		double z0 = (xyb0_b[ix] - xyb0_b[ix2]);
		XybToVals(x0, y0, z0, &valsh0[0], &valsh0[1], &valsh0[2]);
		double x1 = (xyb1_x[ix] - xyb1_x[ix2]);
		double y1 = (xyb1_y[ix] - xyb1_y[ix2]);
		double z1 = (xyb1_b[ix] - xyb1_b[ix2]);
		XybToVals(x1, y1, z1, &valsh1[0], &valsh1[1], &valsh1[2]);
	}
	if (y + 1 < ysize) {
		ix2 = ix + xsize;
	}
	else {
		ix2 = ix - xsize;
	}
	{
		double x0 = (xyb0_x[ix] - xyb0_x[ix2]);
		double y0 = (xyb0_y[ix] - xyb0_y[ix2]);
		double z0 = (xyb0_b[ix] - xyb0_b[ix2]);
		XybToVals(x0, y0, z0, &valsv0[0], &valsv0[1], &valsv0[2]);
		double x1 = (xyb1_x[ix] - xyb1_x[ix2]);
		double y1 = (xyb1_y[ix] - xyb1_y[ix2]);
		double z1 = (xyb1_b[ix] - xyb1_b[ix2]);
		XybToVals(x1, y1, z1, &valsv1[0], &valsv1[1], &valsv1[2]);
	}

	double sup0 = fabs(valsh0[0]) + fabs(valsv0[0]);
	double sup1 = fabs(valsh1[0]) + fabs(valsv1[0]);
	double m = min(sup0, sup1);
	mask_x[ix] = (float)(m);

	sup0 = fabs(valsh0[1]) + fabs(valsv0[1]);
	sup1 = fabs(valsh1[1]) + fabs(valsv1[1]);
	m = min(sup0, sup1);
	mask_y[ix] = (float)(m);

	sup0 = fabs(valsh0[2]) + fabs(valsv0[2]);
	sup1 = fabs(valsh1[2]) + fabs(valsv1[2]);
	m = min(sup0, sup1);
	mask_b[ix] = (float)(m);
}

__kernel void UpsampleSquareRoot(__global float *diffmap, int xsize, int ysize, int step, __global float *diffmap_out)
{
	const int res_x = get_global_id(0);
	const int res_y = get_global_id(1);

	if (res_y + 8 - step >= ysize) return;
	if (res_x + 8 - step >= xsize) return;

	int s2 = (8 - step) / 2;
	// Upsample and take square root.
	const size_t res_xsize = (xsize + step - 1) / step;
	size_t res_ix = (res_y * res_xsize + res_x) / step;
	float orig_val = diffmap[res_ix];
	const float kInitialSlope = 100;
	// TODO(b/29974893): Until that is fixed do not call sqrt on very small
	// numbers.
	double val = orig_val < (1.0 / (kInitialSlope * kInitialSlope))
		? kInitialSlope * orig_val
		: sqrt(orig_val);
	for (size_t off_y = 0; off_y < step; ++off_y) {
		for (size_t off_x = 0; off_x < step; ++off_x) {
			diffmap_out[(res_y + off_y + s2) * xsize +
				res_x + off_x + s2] = val;
		}
	}
}

kernel void CalculateDiffmapGetBlurred(__global float *diffmap, int s, int s2, __global float *blurred)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int xsize = get_global_size(0);
	const int ysize = get_global_size(1);

	blurred[y * xsize + x] = diffmap[(y + s2) * xsize + s + x + s2];
}

kernel void GetDiffmapFromBlurred(__global float *blurred, int s, int s2, __global float *diffmap)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int xsize = get_global_size(0);
	const int ysize = get_global_size(1);

	const double mul1 = 24.8235314874;
	diffmap[(y + s2) * xsize + x + s2]	+= (float)(mul1) * blurred[y * (xsize - s) + x];

}

__kernel void AverageAddImage(__global float *img, __global float *tmp0, __global float *tmp1)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int xsize = get_global_size(0);
	const int ysize = get_global_size(1);

	const int row0 = y * xsize;
	if (x == 0) // excute once per y
	{
		img[row0 + 1] += tmp0[row0];
		img[row0 + 0] += tmp0[row0 + 1];
		img[row0 + 2] += tmp0[row0 + 1];

		img[row0 + xsize - 3] += tmp0[row0 + xsize - 2];
		img[row0 + xsize - 1] += tmp0[row0 + xsize - 2];
		img[row0 + xsize - 2] += tmp0[row0 + xsize - 1];

		if (y > 0) {
			const int rowd1 = row0 - xsize;
			img[rowd1 + 1] += tmp1[row0];
			img[rowd1 + 0] += tmp0[row0];

			img[rowd1 + xsize - 1] += tmp0[row0 + xsize - 1];
			img[rowd1 + xsize - 2] += tmp1[row0 + xsize - 1];
		}
		if (y + 1 < ysize) {
			const int rowu1 = row0 + xsize;
			img[rowu1 + 1] += tmp1[row0];
			img[rowu1 + 0] += tmp0[row0];

			img[rowu1 + xsize - 1] += tmp0[row0 + xsize - 1];
			img[rowu1 + xsize - 2] += tmp1[row0 + xsize - 1];
		}
	}

	if (x >= 2 && x < xsize - 2)
	{
		img[row0 + x - 1] += tmp0[row0 + x];
		img[row0 + x + 1] += tmp0[row0 + x];
	}

	if (x >= 1 && x < xsize - 1) {
		if (y > 0) {
			const int rowd1 = row0 - xsize;
			img[rowd1 + x + 1] += tmp1[row0 + x];
			img[rowd1 + x + 0] += tmp0[row0 + x];
			img[rowd1 + x - 1] += tmp1[row0 + x];
		}
		if (y + 1 < ysize) {
			const int rowu1 = row0 + xsize;
			img[rowu1 + x + 1] += tmp1[row0 + x];
			img[rowu1 + x + 0] += tmp0[row0 + x];
			img[rowu1 + x - 1] += tmp1[row0 + x];
		}
	}
}
