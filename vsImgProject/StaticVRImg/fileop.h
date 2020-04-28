#pragma once
#ifndef _FILEOP_H
#define _FILEOP_H
#include <vector>
#include <string>
#include <unordered_map>
#include <opencv2/core.hpp>
#include "constAndTypes.h"
using ArgType = constandtypes::ArgType;

namespace fileop {
	//typedef std::unordered_map<std::string, bool> cmdType;
	struct ArgList{
		ArgType tool = ArgType::DEFAULT;
		ArgType mode = ArgType::DEFAULT;
	};

	ArgList funTestRead(int argc, const char* argv[], std::vector<std::string>& trainFilePaths, std::vector<std::string>& testFilePaths, const char* keys);
	
	//generate two files: (optionally) recording keypoints and kcenters
	void write_to_file(std::string name, std::vector<cv::KeyPoint>& kpts, cv::Mat& kCenters);
}
#endif