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