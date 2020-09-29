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
			kdTree(cv::Mat source);
			~kdTree();
			std::vector<cv::DMatch> search(cv::Mat& query);
			std::vector<cv::DMatch> colmapSearch(cv::Mat& query);
			cv::Mat getVocab() { return vocab; }
			size_t numWords() { return vocab_size_; }
			void setNN(int NN) { this->numOfNN = NN; }
			static matchOut kdTreeDemo(std::string& img1, std::string& img2, bool display = true);
		private:
			VlKDForest* tree=nullptr;
			size_t vocab_size_;
			int numOfNN;
			cv::Mat vocab;
	};

	std::vector<cv::DMatch> colmapFlannMatcher(const cv::Mat& query_descriptors, const cv::Mat& database_descriptors, int NNeighbors);

	std::vector<cv::DMatch> opencvFlannMatcher(cv::Mat& source, cv::Mat& query);
	void RANSC(cv::Mat& sourceDescrips, std::vector<cv::KeyPoint>& sourcekpts, cv::Mat& queryDescrips, std::vector<cv::KeyPoint>& querykpts, cv::Mat &mask, cv::Mat &homo);
}

#endif
