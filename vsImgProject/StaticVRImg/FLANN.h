#pragma once
#ifndef _FLANN_H
#define _FLANN_H
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
namespace FLANN {
	void FLANNMatch(cv::Mat& descripts1, cv::Mat& descripts2, std::vector<cv::DMatch>& good_matches);
	void FLANNImgsMatch(cv::Mat& grayImg1, cv::Mat& grayImg2, std::vector<cv::DMatch>& good_matches);
	double FLANNScore(std::vector<cv::DMatch>& good_matches);
}

#endif // !_FLANN_H
