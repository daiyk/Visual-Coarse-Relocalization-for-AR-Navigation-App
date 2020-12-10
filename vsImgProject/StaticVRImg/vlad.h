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
		vlad(std::vector<std::string>& paths); //training and writing the centers and encoder
		vlad(std::string centerPath, std::string encPath); //read the centers and encoder 
		vlad(std::vector<cv::Mat> descripts); //train the vlad center and encoding vector for the given database descriptors
		~vlad();
		void search(cv::Mat img, std::vector<int>& ind, std::vector<double>& score, int bestOfAll); //img must be grayImg
		void searchWithDescripts(cv::Mat img, std::vector<int>& ind, std::vector<double>& score, int bestOfAll);
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
