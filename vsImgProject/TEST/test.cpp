#include "StaticVRImg/graph.h"
#include "StaticVRImg/extractor.h"
#include "StaticVRImg/fileManager.h"
#include "StaticVRImg/matcher.h"
#include <opencv2/core.hpp>
#include <igraph.h>
#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <nlohmann/json.hpp>

int graphbuildTest(int argc, const char* argv[]) {
	//graph::graphTest();
	//test 
	//testing the build function with graph
	if (argc < 3) {
		std::cout << "Please provides path to the dictionary and image!" << std::endl;
	}
	cv::FileStorage reader;
	reader.open(argv[1],cv::FileStorage::READ);
	if (!reader.isOpened()) { std::cout << "failed to open the kcenter file" << std::endl; }
	//read kcenters
	cv::Mat kCenters;
	reader["kcenters"] >> kCenters;

	//do matching on the test image 
	std::string testImg(argv[2]);
	std::vector<std::string> trainPath;
	trainPath.push_back(testImg);
	cv::Mat descripts;
	std::vector<cv::KeyPoint> kpts;
	extractor::vlimg_descips_compute(trainPath, descripts, kpts);

	//do kd-tree on the source descriptors
	std::vector<cv::DMatch> matches = matcher::kdTree(kCenters, descripts);
	igraph_t mygraph;
	bool status = graph::build(matches, kpts, mygraph);
	if (!status) { std::cout << "graph build failed! check your function." << std::endl; return 0; }

	fileManager::write_graph(mygraph, "MatchingTest", "graphml");
}

void readUserTest() {
	std::string usersetting = "D:\\thesis\\Visual-Coarse-Relocalization-for-AR-Navigation-App\\User\\vrn_set.json";
	fs::path userset(usersetting);
	json testList =  fileManager::read_user_set(userset);
	std::string dumprel = testList.dump();
	std::cout << dumprel << std::endl;

	std::cout << "read the actual params"<<fileManager::parameters::maxNumDeg << std::endl << fileManager::parameters::numOfAttemp << std::endl;

}
int main(int argc, const char* argv[]) {
	readUserTest();
}



//Eigen::MatrixXi test(9, 2);
//test.col(0) << 1, 5, 6, 7, 9, 0, 10, 20, 11;
//test.col(1) << 3, 4, 6, 7, 8, 9, 0, 1, 5;
//Eigen::MatrixXi index(9, 2);
//int x = 0;
//std::iota(index.col(0).data(), index.col(0).data() + index.col(0).size(), x++);
//int y = 0;
//std::iota(index.col(1).data(), index.col(1).data() + 9, y++);
//std::sort(index.col(0).data(), index.col(0).data() + 9, [&](int left, int right) {return test.col(0)(left) < test.col(0)(right); });
//
//std::cout << "Orig :" << test << std::endl << " indexs: ";
//std::cout << index;
//return 0;