#pragma once
#ifndef _FILEMANAGER_H
#define _FILEMANAGER_H
#include <string>
#include <vector>
#include <iostream>
#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>
#include <FreeImage.h>
#define IGRAPH_STATIC 1
#include "igraph.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;
namespace fs = boost::filesystem;
namespace fileManager {
	class BaseFile {
		virtual bool Read(std::ifstream& instream) const = 0;
		virtual bool Write(json root) const = 0;
	};
	class graphManager : public BaseFile {
	public:
		graphManager(std::string& rootPath) { root_path_ = rootPath; }
		bool writeGraph(igraph_t mygraph);
		void setName(std::string name) { graph_name_ = name; }
		bool readGraph(std::string path);
		bool writeEdges(const std::vector<int> &edges);
		bool writeLabels(const std::vector<int> &labels);
		bool writeWeights(const std::vector<int> &weights);
	private:
		bool Read(std::ifstream& instream){}
		bool Write(json root){}
		std::string graph_name_="";
		std::string root_path_;
		json iobuffer_;
	};
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
		static std::string userSetPath; //path to the usersetting file path
		static int octave;       // number of octave used in the sift detection
		static int noctaveLayer; // scale layers per octave
		static int firstOctaveInd; // start from 1th octave, which means -1 octave that double the original image size
		static double sigma_0; // sigma for the #0 octave
		static int centers;    // k-means center detection, defines the number of centers
		static int numOfAttemp; //times of try to compute the center for each cluster, five times to choose the best one
		static int numOfItera;
		static int descriptDim; //descriptDim default to SIFT 128
		static double accuracy;
		static double siftEdgeThres;
		static double siftPeakThres;
		static double imgScale;
		static int maxNumOrient; //max number orientation extracted by vlfeat covdet feature detector
		static int maxNumFeatures; //max number of features allowed in the detection
		static std::string tfidfPath; //path to the tfidf score file

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

	struct covisOptions {
		// Root path to folder which contains the images.
		std::string image_path = "";

		// list of images that are read from the path. The list must contain the relative path
		std::vector<std::string> image_list;
		
		//dictionary path
		std::string vocab_path = "";

		//rgb
		bool rgb = false;

		//database path
		std::string database_path = "";

		//num of inlier images to keep
		int numImageToKeep = 2;
	};
	
}

#endif // 
