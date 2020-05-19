#pragma once
#ifndef _FILEMANAGER_H
#define _FILEMANAGER_H
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/core.hpp>
#define IGRAPH_STATIC 1
#include "igraph.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;
namespace fs = std::filesystem;
namespace fileManager {
	enum class ArgType {
		TOOL_OPENCV,
		TOOL_VLFEAT,
		TOOL_OPENCV_AND_VLFEAT,
		MODE_TRAIN,
		MODE_MATCHING,
		MODE_DEMO,
		DEFAULT
	};

	struct ArgList {
		ArgType tool;
		ArgType mode;
		ArgList() :tool(ArgType::DEFAULT), mode(ArgType::DEFAULT) {}
	};

	//parameters list that will be updated with file input
	struct parameters {
		// below will be transformed to json file.
		static int octave;       // number of octave used in the sift detection
		static int noctaveLayer; // scale layers per octave
		static int octave_start; // learning start from 1th octave, -1 for more details
		static double sigma_0; // sigma for the #0 octave
		static int centers;    // k-means center detection, defines the number of centers
		static int numOfAttemp; //times of try to compute the center for each cluster, five times to choose the best one
		static int numOfItera;
		static double accuracy;

		//OpenCV relevent setting
		static cv::TermCriteria criteria; //stop criteria, COUNT means number of iter, EPS means convergence accuracy
		static float MATCH_THRES; //define the threshold for matching 

		//matching setting
		static int numOfNN;

		static size_t maxNumDeg;
		static double radDegLim; //the radius limits for edge connection, limits the edge < radDegLim*pts.scale
	};
	ArgList funTestRead(int argc, const char* argv[], std::vector<std::string>& trainFilePaths, std::vector<std::string>& testFilePaths, const char* keys);
	void write_to_file(std::string name, std::vector<cv::KeyPoint>& kpts, cv::Mat& kCenters);
	void write_graph(igraph_t& graph, std::string name, std::string mode);
	json read_user_set(fs::path& params);
}
#endif // 
