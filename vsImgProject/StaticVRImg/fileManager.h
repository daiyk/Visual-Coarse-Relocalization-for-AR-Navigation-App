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
		static int firstOctaveInd; // start from 1th octave, which means -1 octave that double the original image size
		static double sigma_0; // sigma for the #0 octave
		static int centers;    // k-means center detection, defines the number of centers
		static int numOfAttemp; //times of try to compute the center for each cluster, five times to choose the best one
		static int numOfItera;
		static int descriptDim; //descriptDim default to SIFT 128
		static double accuracy;
		static double distRat;
		static double siftEdgeThres;
		static double siftPeakThres;
		static double imgScale;
		

		//OpenCV relevent setting
		static cv::TermCriteria criteria; //stop criteria, COUNT means number of iter, EPS means convergence accuracy
		static float MATCH_THRES; //define the threshold for matching 

		//matching setting
		static int numOfNN; // number of nearest neighbors extracted from kdtree comparison 
		static int maxNumComp; // maximum number of comparison during kdtree searching

		//graph building relevent setting
		static size_t maxNumDeg;
		static double radDegLim; //the radius limits for edge connection, limits the edge < radDegLim*pts.scale
	};
	ArgList funTestRead(int argc, const char* argv[], std::vector<std::string>& trainFilePaths, std::vector<std::string>& testFilePaths, const char* keys);
	void write_to_file(std::string name, std::vector<cv::KeyPoint>& kpts, cv::Mat& kCenters);
	void write_graph(igraph_t& graph, std::string name, std::string mode);
	void read_user_set(fs::path& params);
}
#endif // 
