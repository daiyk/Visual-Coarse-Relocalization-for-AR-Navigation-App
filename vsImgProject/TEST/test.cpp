#include "StaticVRImg/graph.h"
#include "StaticVRImg/extractor.h"
#include "StaticVRImg/fileManager.h"
#include "StaticVRImg/matcher.h"
#include "StaticVRImg/kernel.h"
#include <opencv2/core.hpp>
#include <filesystem>
#include <igraph.h>
#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <nlohmann/json.hpp>
void filecreate() {
	std::filesystem::path curPath = std::filesystem::current_path();
	std::filesystem::path resultPath;
	for (auto it : std::filesystem::recursive_directory_iterator(curPath)) {
		if (it.path().stem().string() == "vsImgProject") {
			resultPath = it.path();
			break;
		}
	}
	std::filesystem::create_directory(resultPath.parent_path() / "Result");
}
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
	fileManager::read_user_set(userset);
	/*std::string dumprel = testList.dump();
	std::cout << dumprel << std::endl;

	std::cout << "read the actual params"<<fileManager::parameters::maxNumDeg << std::endl << fileManager::parameters::numOfAttemp << std::endl;*/

}

void graphKernelTest() {
	
	auto sTime = clock();
	igraph_t testGraph;
	igraph_empty(&testGraph, 6, IGRAPH_UNDIRECTED);
	igraph_vector_t e;
	igraph_real_t edges[] = { 0,1,1,4,1,5,1,2,2,3,3,4,4,5 };
	igraph_vector_view(&e, edges, sizeof(edges) / sizeof(double));
	igraph_create(&testGraph, &e, 0, IGRAPH_UNDIRECTED);

	//add labels
	std::vector<int> labstd{ 2,1,3,1,5,1 };
	std::vector<double> labstd_vec(labstd.begin(), labstd.end());
	igraph_vector_t lab_vec;
	
	igraph_vector_view(&lab_vec, labstd_vec.data(), 6);

	igraph_cattribute_VAN_setv(&testGraph, "label", &lab_vec);

	//defines the kernel class
	kernel::robustKernel kernelObj(2, 6);
	kernelObj.push_back(testGraph);
	kernelObj.push_back(testGraph);
	auto k_matrix = kernelObj.robustKernelCom();

	std::cout << " -> igraph spends times" << (clock() - sTime) / double(CLOCKS_PER_SEC) << " secs......" << std::endl;

	int cols = igraph_matrix_ncol(&k_matrix);
	int rows = igraph_matrix_nrow(&k_matrix);
	for (int i = 0; i < cols; i++) {
		std::cout << std::endl;
		for (int j = 0; j < rows; j++) {
			std::cout << MATRIX(k_matrix, i, j)<<"\t";
		}
	}
	


	//test with old functions
	sTime = clock();
	Eigen::MatrixXi E(7,3);
	Eigen::MatrixXd k_mat;
	E << 0, 1, 1, 1, 4, 1, 1, 5, 1, 1, 2, 1, 2, 3, 1, 3, 4, 1, 4, 5, 1;
	
	std::vector<int> num_v{6,6};
	std::vector<int> num_e{ 7,7 };
	int h_max = 2;
	std::vector<std::vector<int>> labsstd;
	labsstd.push_back(labstd);
	labsstd.push_back(labstd);
	std::vector<Eigen::MatrixXi> Es;
	Es.push_back(E);
	Es.push_back(E);

	
	kernel::robustKernel::wlRobustKernel(Es, labsstd, num_v, num_e, h_max, k_mat);

	std::cout << " -> old funcs spends times" << (clock() - sTime) / double(CLOCKS_PER_SEC) << " secs......" << std::endl;
	std::cout << std::endl<<k_mat;
}

void graphBuildPlusKernelTest(int argc, const char* argv[]) {
	if (argc < 3) {
		std::cout << "Please provides path to the dictionary and image!" << std::endl;
	}

	//read and build graph for the first test image
	cv::FileStorage reader;
	reader.open(argv[1], cv::FileStorage::READ);
	if (!reader.isOpened()) { std::cout << "failed to open the kcenter file" << std::endl; }
	//read kcenters
	cv::Mat kCenters;
	reader["kcenters"] >> kCenters;
	reader.release();
	//do matching on the two test image 
	std::string testImg1(argv[2]);
	std::vector<std::string> trainPath;
	trainPath.push_back(testImg1);
	cv::Mat descripts1;
	std::vector<cv::KeyPoint> kpts1;
	extractor::vlimg_descips_compute(trainPath, descripts1, kpts1);
	//do kd-tree on the source descriptors
	std::vector<cv::DMatch> matches = matcher::kdTree(kCenters, descripts1);
	igraph_t mygraph1;
	bool status = graph::build(matches, kpts1, mygraph1);
	if (!status) { std::cout << "graph build failed! check your function." << std::endl;}

	fileManager::write_graph(mygraph1, "testimg1", "graphml");
	trainPath.pop_back();


	//read and build graph for the second test image
	//do matching on the two test image 
	std::string testImg2(argv[3]);
	trainPath.push_back(testImg2);
	cv::Mat descripts2;
	std::vector<cv::KeyPoint> kpts2;
	extractor::vlimg_descips_compute(trainPath, descripts2, kpts2);
	//do kd-tree on the source descriptors
	matches = matcher::kdTree(kCenters, descripts2);
	igraph_t mygraph2;
	status = graph::build(matches, kpts2, mygraph2);
	if (!status) { std::cout << "graph build failed! check your function." << std::endl; }
	fileManager::write_graph(mygraph2, "testimg2", "graphml");

	auto sTime = clock();

	//add to the graph bags and compute the matching score
	kernel::robustKernel kernelObj(2, kCenters.rows);
	kernelObj.push_back(mygraph1);
	kernelObj.push_back(mygraph2);

	std::cout << " -> igraph spends times" << (clock() - sTime) / double(CLOCKS_PER_SEC) << " secs......" << std::endl;
	auto resMat = kernelObj.robustKernelCom();
	int cols = igraph_matrix_ncol(&resMat);
	int rows = igraph_matrix_nrow(&resMat);
	for (int i = 0; i < cols; i++) {
		std::cout << std::endl;
		for (int j = 0; j < rows; j++) {
			std::cout << MATRIX(resMat, i, j) << "\t";
		}
	}

}

int main(int argc, const char* argv[]) {
	igraph_i_set_attribute_table(&igraph_cattribute_table);
	graphBuildPlusKernelTest(argc, argv);
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