#pragma once
#ifndef _HELPER_H
#define _HELPER_H
#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS
#include <opencv2/core.hpp>
#include <unordered_set>
#include <ctime>
#include <random>
#include <colmap/feature/types.h>
namespace helper {
	//helper function to get current date
	std::string dateTime();

	//unpack octave number, copy from opencv source code https://github.com/opencv/opencv_contrib/blob/bebfd717485c79644a49ac406b0d5f717b881aeb/modules/xfeatures2d/src/sift.cpp#L214-L220
	void unpackOctave(const cv::KeyPoint& kpt, int& octave, int& layer, float& scale);

	//tolerence comparison of two double numbers
	bool isEqual(double x, double y);

	//transform descriptor from float to unsigned bits
	cv::Mat DescriptorFloatToUint(cv::Mat descripts);

	//truncate features and reserve top scales features
	void ExtractTopFeatures(colmap::FeatureKeypoints* keypoints, colmap::FeatureVisualIDs* ids, const size_t num_features);


	//1th score functions: multiplicative
	void computeScore1(std::vector<std::vector<double>>& raw_scores, std::vector<size_t>& edge_nums, std::vector<double>& raw_self_scores, bool tfidfWeight = false);
	
	//2th score function: discriminative query score based
	void computeScore2(std::vector<std::vector<double>>& raw_scores, std::vector<size_t>& edge_nums, std::vector<double>& raw_self_scores);

	//3th score computing function
	void computeScore3(std::vector<std::vector<double>>& raw_scores, std::vector<double>& raw_self_scores);

	//random pick up n elements
	std::unordered_set<int> pickSet(int N, int k, std::mt19937& gen);
	inline std::string cvtype2str(int type) {
		std::string r;

		uchar depth = type & CV_MAT_DEPTH_MASK;
		uchar chans = 1 + (type >> CV_CN_SHIFT);

		switch (depth) {
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
		}

		r += "C";
		r += (chans + '0');

		return r;
	}
}
#endif
