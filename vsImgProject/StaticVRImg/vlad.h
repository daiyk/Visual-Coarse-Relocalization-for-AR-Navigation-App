#pragma once
#ifndef _VLAD_H
#define _VLAD_H
#include <vector>
#include <string>
#include <opencv2/core.hpp>

extern "C" {
	#include "vl/vlad.h"
	#include "vl/kdtree.h"
}

namespace vlad {
	class vlad {
	public:
		vlad(std::vector<std::string>& paths); //training and writing the centers, encoder
		vlad(std::string centerPath, std::string encPath); //read the centers and encoder 
		void search(cv::Mat img, std::vector<int>& ind, std::vector<double>& score, int bestOfAll); //img must be grayImg
		cv::Mat1f& getEncoder() { return this->enc; }
	private:
		void write_to_file(std::string name);
		void enc_index(cv::Mat1f& query, std::vector<int>& ind, std::vector<double>& score, int bestOfAll);
		VlKDForest* tree;
		cv::Mat1f enc;
		cv::Mat kCenters;
	};
}
#endif // !_VLAD_H
