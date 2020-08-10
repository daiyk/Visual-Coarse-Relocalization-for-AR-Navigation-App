#pragma once
#ifndef _HELPER_H
#define _HELPER_H
#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS
#include <opencv2/core.hpp>
#include <ctime>
namespace helper {
	//helper function to get current date
	std::string dateTime();

	//unpack octave number, copy from opencv source code https://github.com/opencv/opencv_contrib/blob/bebfd717485c79644a49ac406b0d5f717b881aeb/modules/xfeatures2d/src/sift.cpp#L214-L220
	void unpackOctave(const cv::KeyPoint& kpt, int& octave, int& layer, float& scale);

	//tolerence comparison of two double numbers
	bool isEqual(double x, double y);

	//1th score functions: multiplicative
	void computeScore1(std::vector<std::vector<double>>& raw_scores, std::vector<size_t>& edge_nums, std::vector<double>& raw_self_scores, bool tfidfWeight = false);
	
	//2th score function: discriminative query score based
	void computeScore2(std::vector<std::vector<double>>& raw_scores, std::vector<size_t>& edge_nums, std::vector<double>& raw_self_scores);
}
#endif
