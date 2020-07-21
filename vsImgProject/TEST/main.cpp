#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <Eigen/Core>
#define IGRAPH_STATIC 1
#include <igraph.h>
#include "UKsets.h"
#include "Graphtest.h"
#include "StaticVRImg/fileManager.h"
#include "StaticVRImg/matcher.h"
#include "StaticVRImg/graph.h"
#include "StaticVRImg/vlad.h"
using params = fileManager::parameters;
std::string type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

int main(int argc, const char* argv[]) {
	if (argc < 2) {
		std::cout << "please provides the path to UKBench imagesets" << std::endl;
		//return 0;
	}
	fileManager::read_user_set();
	std::cout << "center number: " << params::centers << std::endl;

	if (argc == 2) {
		/*std::vector<std::string> paths;
		fileManager::read_files_in_path(argv[1], paths);
		vlad::vlad encoder(paths);
		auto enc = encoder.getEncoder();
		std::cout << "size is: " << enc.size()<<std::endl;*/
		UKB::UKFLANNTest(argc, argv, params::sampleSize, params::imgsetSize);
	}
	if (argc == 3) {
		UKB::UKtest(argc, argv, params::sampleSize, params::imgsetSize);
		//test two imgs
		/*graphBuildPlusKernelTest(argc, argv);*/
	}
	else if(argc==4)
	{
		graphBuildPlusKernelTest(argc, argv);
		/*graphbuildTest(argc, argv);*/
		/*UKB::UKVladTest(argc, argv, params::sampleSize, params::imgsetSize);*/
	}
	return 0;

}