#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <Eigen/Core>
#include "UKsets.h"
#include "Graphtest.h"
#include "StaticVRImg/fileManager.h"
#include "StaticVRImg/matcher.h"
#include "StaticVRImg/graph.h"
void main(int argc, const char* argv[]) {
	if (argc < 2) {
		std::cout << "please provides the path to UKBench imagesets" << std::endl;
	}
	fileManager::read_user_set();
	std::cout << "center number: :" << fileManager::parameters::centers << std::endl;


	//read kcenters
	//cv::FileStorage reader;
	//reader.open(argv[1], cv::FileStorage::READ);
	//if (!reader.isOpened()) { std::cout << "failed to open the kcenter file" << std::endl; return; }
	////read kcenters
	//cv::Mat kCenters;
	//reader["kcenters"] >> kCenters;
	//reader.release();
	//std::vector<std::string> trainPath;
	//trainPath.push_back("D:\\thesis\\ukbench\\full\\ukbench00082.jpg");
	//cv::Mat descripts1;
	//std::vector<cv::KeyPoint> kpts1;
	//try {
	//	extractor::vlimg_descips_compute(trainPath, descripts1, kpts1);
	//}
	//catch (std::invalid_argument& e) {
	//	std::cout << e.what() << std::endl;
	//};
	//std::vector<cv::DMatch> matches = matcher::kdTree(kCenters, descripts1);
	//igraph_t mygraph1;
	//bool status = graph::build(matches, kpts1, mygraph1);

	//need to test the model for picture

	/*UKB::UKtrain(argc, argv, 20);*/
	UKB::UKtest(argc, argv, 2, 20);
	/*graphBuildPlusKernelTest(argc, argv);*/

	return;
	
}