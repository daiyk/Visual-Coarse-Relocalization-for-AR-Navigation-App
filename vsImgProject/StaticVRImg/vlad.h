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
		vlad(std::vector<std::string>& paths, int numOfCenter);
		int search(cv::Mat img);
	private:
		VlKDForest* tree;
		std::vector<float> enc;
		cv::Mat kCenters;
	};
}
#endif // !_VLAD_H
