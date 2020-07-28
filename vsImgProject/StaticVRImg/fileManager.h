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
	//default user_set location
	extern fs::path user_set_default;

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
		bool homo;
		ArgList() :tool(ArgType::DEFAULT), mode(ArgType::DEFAULT),homo(false) {}
	};

	//parameters list that will be updated with file input
	struct parameters {

		/******* all parameters below will be read from json file***********/
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
		static int maxNumOrient; //max number orientation extracted by vlfeat covdet feature detector
		

		//OpenCV relevent setting
		static cv::TermCriteria criteria; //stop criteria, COUNT means number of iter, EPS means convergence accuracy
		static float MATCH_THRES; //define the threshold for matching 

		//matching setting
		static int numOfNN; // number of nearest neighbors extracted from kdtree comparison 
		static int maxNumComp; // maximum number of comparison during kdtree searching

		//vlad setting 
		static int vladCenters; //vlad.h: kmeans center number

		//graph building relevent setting
		static size_t maxNumDeg; //graph.h: max num of deg per node allowed
		static double radDegLim; //graph.h: the radius limits for edge connection, limits the edge < radDegLim*pts.scale

		//testing function relative setting
		static int sampleSize; //TEST: num of sampled imgs
		static int imgsetSize; //TEST: total image database size

		//covisMap setting
		static double PCommonwords; //kernel.h: Minimum percentage of virtual words overlapped with query required to select a given clique for virtual locations
		static double PCliques; //kernel.h: Minimum percentage of virtual words overlapped between cliques to extend these cliques
	};
	ArgList funTestRead(int argc, const char* argv[], std::vector<std::string>& trainFilePaths, std::vector<std::string>& testFilePaths, const char* keys);
	
	void read_files_in_path(std::string path, std::vector<std::string> &paths);
	void write_to_file(std::string name, std::vector<cv::KeyPoint>& kpts, cv::Mat& kCenters);
	void write_graph(igraph_t& graph, std::string name, std::string mode);
	void read_user_set(fs::path params= user_set_default);
}
#endif // 
