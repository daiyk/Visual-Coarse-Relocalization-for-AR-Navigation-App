#pragma once

#ifndef _MATCHER_H
#define _MATCHER_H
#include <vector>
#include <string>
#include <opencv2/core.hpp>


namespace matcher {
	struct matchOut {
		std::vector<cv::DMatch> matches;
		std::vector<cv::KeyPoint> source;
		std::vector<cv::KeyPoint> refer;
	};
	matchOut kdTree(std::string& img1, std::string& img2, bool display=true);
}

#endif
