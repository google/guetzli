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

float calcWeight(__global float* multipliers, int len)
{
	float weight_no_border = 0;
	for (int j = 0; j < len; j++)
	{
		weight_no_border += multipliers[j];
	}
	return weight_no_border;
}

__kernel void Convolution(__global float* multipliers, __global float* inp, __global float* result,
							int xsize, int xstep, int len, int offset, float border_ratio)
{
	const int ox = get_global_id(0);
	const int y = get_global_id(1);

	const int oxsize = get_global_size(0);
	const int ysize = get_global_size(1);

	const int x = ox * xstep;
/*
	float weight_no_border = 0;
	for (int j = 0; j <= 2 * offset; j++)
	{
		weight_no_border += multipliers[j];
	}
*/
	float weight_no_border = calcWeight(multipliers, len);

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

	const int sample_x = x / xstep * xstep;
	const int sample_y = y / ystep * ystep;

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

__kernel void ScaleImage(double scale, __global float *result)
{
	const int i = get_global_id(0);
	result[i] *= (float)(scale);
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
	int res_xsize,
	__global float *result)
{
	const int res_x = get_global_id(0);
	const int res_y = get_global_id(1);

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

inline double Interpolate(const double *array, int size, double sx) {
	double ix = fabs(sx);

	int baseix = static_cast<int>(ix);
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

std::array<double, 21> MakeLowFreqColorDiffDy() {
	std::array<double, 21> lut;
	static const double inc = 5.2511644570349185;
	lut[0] = 0.0;
	for (int i = 1; i < 21; ++i) {
		lut[i] = lut[i - 1] + inc;
	}
	return lut;
}

const double *GetLowFreqColorDiffDy() {
	static const std::array<double, 21> kLut = MakeLowFreqColorDiffDy();
	return kLut.data();
}

void XybLowFreqToVals(double x, double y, double z,
	double *valx, double *valy, double *valz) {
	static const double xmul = 6.64482198135;
	static const double ymul = 0.837846224276;
	static const double zmul = 7.34905756986;
	static const double y_to_z_mul = 0.0812519812628;
	z += y_to_z_mul * y;
	*valz = z * zmul;
	*valx = x * xmul;
	*valy = Interpolate(GetLowFreqColorDiffDy(), 21, y * ymul);
}

void XybDiffLowFreqSquaredAccumulate(double r0, double g0, double b0,
	double r1, double g1, double b1,
	double factor, double res[3]) {
	double valx0, valy0, valz0;
	double valx1, valy1, valz1;
	XybLowFreqToVals(r0, g0, b0, &valx0, &valy0, &valz0);
	if (r1 == 0.0 && g1 == 0.0 && b1 == 0.0) {
		PROFILER_ZONE("XybDiff r1=g1=b1=0");
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

__kernel void edgeDetectorMap(__global float *result, __global float *r, __global float *g, __global float* b, __global float *r2, __global float* g2, __global float *b2, int xsize, int ysize, int step)
{
	const int result_x = get_global_id(0);
	const int result_y = get_global_id(1);

	const int result_xsize = get_global_size(0);
	const int result_ysize = get_global_size(1);

	int pos_x = result_x * step;
	int pos_y = result_y * step;

	int local_count = 0;
	double local_xyb[3] = { 0 };
	const double w = 0.711100840192;

	int offset[4][2] = { { 0£¬0}£¬ { 0£¬7}£¬{ 7£¬0}£¬{ 7£¬7} };
	int edgeSize = 3;

	for (int k = 0; i < 4; k++)
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

	static const double weight = 0.01617112696;
	const double mul = weight * 8.0 / local_count;

	int idx = (result_y * result_xsize + result_x) * 3;
	result[idx]     = local_xyb[0];
	result[idx + 1] = local_xyb[1];
	result[idx + 2] = local_xyb[2];
}
