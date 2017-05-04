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

__kernel void DownSample(__global float* pA, __global float* pC, int square)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int xsize = get_global_size(0);
	const int ysize = get_global_size(1);

	const int oxsize = xsize / square;

	const int sample_x = x / square;
	const int sample_y = y / square;

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

float Gamma(double v)
{
	double min_value = 0.770000000000000;
	double max_value = 274.579999999999984;

	/*static*/ const double p[5 + 1] = {
		881.979476556478289, 1496.058452015812463, 908.662212739659481,
		373.566100223287378, 85.840860336314364, 6.683258861509244,
	};
	/*static*/ const double q[5 + 1] = {
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
	/*static*/ const double a0 = 1.01611726948;
	/*static*/ const double a1 = 0.982482243696;
	/*static*/ const double a2 = 1.43571362627;
	/*static*/ const double a3 = 0.896039849412;
	*valx = a0 * r - a1 * g;
	*valy = a2 * r + a3 * g;
	*valz = b;
}

__kernel void OpsinDynamicsImage(__global float *r, __global float *g, __global float *b, __global float *r_blurred, __global float *g_blurred, __global float *b_blurred, int size)
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