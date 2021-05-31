#include "stdafx.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

bool isInside(Mat img, int i, int j)
{
	if (i >= 0 && i < img.rows && j >= 0 && j < img.cols)
		return true;
	return false;
}

long long census_transform(Mat_<uchar> &img, int i, int j)
{
	//long long - 64 bits
	//9x7 = 54 < 64
	long long bitarray = 0;
	int di[7] = { -3, -2, -1, 0, 1, 2, 3 };
	int dj[9] = { -4, -3, -2, -1, 0, 1, 2, 3, 4 };

	for (int u = 0; u < 7; u++)
	{
		for (int v = 0; v < 9; v++)
		{
			if (isInside(img, i + di[u], j + dj[v]))
			{
				if (!(u == (i + di[u]) && v == (j + dj[v])))
				{
					if (img(i + di[u], j + dj[v]) < img(i, j))
						bitarray += 1;
					else
						bitarray += 0;
					bitarray <<= 1;

				}
			}
		}
	}
	return bitarray;
}

int hamming_distance(long long bits_right, long long bits_left) 
{
	int different_bits = 0;

	long long n = bits_left ^ bits_right;//XOR
	int nb = 63;
	while (nb--) 
	{
		different_bits += n & 1;
		n >>= 1;
	}

	return different_bits;
}

bool checkBoundaries(Mat img_L, Mat img_R, int i, int j, int d, int census_height, int census_width)
{
	if (i >= census_height && i <= img_L.rows - census_height)
		if(j >= census_width && j <= img_L.cols - census_width)
			if(j - d >= census_width && j - d <= img_R.cols - census_width)
				return true;
	return false;
}

Mat_<int> compute_cost(Mat_<uchar> img_R, Mat_<uchar> img_L)
{
	int census_height = 7;
	int census_width = 9;
	int dimension[] = { img_R.rows, img_R.cols, img_R.cols };

	Mat_<int> C = Mat(3, dimension, CV_8UC1);

	for (int i = 0; i < img_L.rows; i++)
	{
		for (int j = 0; j < img_L.cols; j++)
		{
			for (int d = 1; d <= 50; d++)
			{
				if (checkBoundaries(img_L, img_R, i, j, d, census_height, census_width))
					//j - d -> right image shifts horizontally from right to left
					C(i, j, d) = hamming_distance(census_transform(img_L, i, j), census_transform(img_R, i, j - d));
				else
					C(i, j, d) = 0;
			}
		}
	}

	return C;

}
Mat_<uchar> compute_disparity_map(Mat_<uchar> img_R, Mat_<uchar> img_L)
{
	int dimension[] = { img_R.rows, img_R.cols, img_R.cols };
	Mat_<int> C = Mat(3, dimension, CV_8UC1);
	Mat_<int> S = Mat(3, dimension, CV_8UC1);//sum of hamming distances from the DSI in a window of 100 (10x10)

	Mat_<uchar> disparity_map(img_L.rows, img_L.cols);
	int census_height = 7;
	int census_width = 9;

	C = compute_cost(img_R, img_L);

	for (int i = census_height; i < img_L.rows - census_height; i++)
	{
		for (int j = census_width; j < img_L.cols - census_width; j++) 
		{
			int min_sum = INT_MAX;
			int min_disperity = 0;

			for (int d = 1; d <= 50; d++) 
			{
				int sum = 0;
				for (int u = -5; u < 5; u++) 
					for (int v = -5; v < 5; v++) 
						sum += C(i + u, j + v, d);

				S(i, j, d) = sum;
				if (sum < min_sum)
				{
					min_sum = sum;
					min_disperity = d;
				}
			}
			disparity_map(i, j) = min_disperity;
		}
	}
	return disparity_map;
}

Mat_<uchar> median_filter(Mat_<uchar> img, int w)
{
	Mat_<uchar> dst(img.rows, img.cols);
	int* orderedStatistic = new int[w * w];

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int nr_elems = 0;
			for (int u = 0; u < w; u++)
			{
				for (int v = 0; v < w; v++)
				{
					if (isInside(img, i + u - w / 2, j + v - w / 2))
					{
						nr_elems++;
						orderedStatistic[nr_elems] = img(i + u - w / 2, j + v - w / 2);
					}
				}
			}
			std::sort(orderedStatistic, orderedStatistic + nr_elems);
			dst(i, j) = orderedStatistic[nr_elems / 2];
		}
	}
	return dst;
}

float calculate_error(Mat_<uchar> img, Mat_<uchar> result) 
{
	float d = 0;
	float n = img.rows * img.cols;
	for (int i = 0; i < img.rows; i++) 
		for (int j = 0; j < img.cols; j++) 
			if (abs(img(i, j) - result(i, j)) > 2) //value set = 2
				d++;

	return d / n;
}

int main() 
{
	Mat_<uchar> img_R = imread("Images/teddyimR.png", 0);
	Mat_<uchar> img_L = imread("Images/teddyimL.png", 0);
	Mat_<uchar> imgGT = imread("Images/teddyDispL.png", 0);

	imshow("Image right", img_R);
	imshow("Image left", img_L);
	imshow("Ground truth", imgGT);

	Mat_<uchar> disparity_map(img_R.rows, img_R.cols);
	Mat_<uchar> disparity_map_Filtered(img_R.rows, img_R.cols);

	disparity_map = compute_disparity_map(img_R, img_L);
	//imshow("Disparity map", disparity_map * 4);
	imwrite("Disparity_map.bmp", disparity_map * 4);

	int w = 10;
	disparity_map_Filtered = median_filter(disparity_map, w);
	//imshow("Disparity map filtered", disparity_map_Filtered * 4);
	imwrite("Disparity_map_filtered.bmp", disparity_map_Filtered * 4);

	printf("Error with filtering: %f\n", calculate_error(disparity_map_Filtered * 4, imgGT));
	printf("Error: %f\n", calculate_error(disparity_map * 4, imgGT));

	waitKey(0);

	return 0;
}