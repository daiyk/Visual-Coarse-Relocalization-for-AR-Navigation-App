#pragma once

#ifndef _MATCHER_H
#define _MATCHER_H
#include <vector>
#include <string>
#include <opencv2/core.hpp>
extern "C" {
	#include "vl/kdtree.h"
}


namespace matcher {
	struct matchOut {
		std::vector<cv::DMatch> matches;
		std::vector<cv::KeyPoint> source;
		std::vector<cv::KeyPoint> refer;
	};
	matchOut kdTreeDemo(std::string& img1, std::string& img2, bool display=true);
	std::vector<cv::DMatch> kdTree(cv::Mat &source, cv::Mat &query);

	class vlad {
	public:
		vlad(std::vector<std::string>& paths, int numOfCenter);
		int search(std::vector<std::string>& queryImgs);
	private:
		VlKDForest* tree;
		std::vector<float> enc;
		cv::Mat kCenters;
	};
}

#endif
