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

	class kdTree {
		public:
			kdTree(cv::Mat& source);
			~kdTree();
			std::vector<cv::DMatch> search(cv::Mat& query);
			static matchOut kdTreeDemo(std::string& img1, std::string& img2, bool display = true);
		private:
			VlKDForest* tree=nullptr;
	};
	
	std::vector<cv::DMatch> opencvFlannMatcher(cv::Mat& source, cv::Mat& query);
}

#endif
