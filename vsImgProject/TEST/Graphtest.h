#include "StaticVRImg/graph.h"
#include "StaticVRImg/extractor.h"
#include "StaticVRImg/fileManager.h"
#include "StaticVRImg/matcher.h"
#include "StaticVRImg/kernel.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <igraph.h>
#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <random>
#include <numeric>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include "StaticVRImg/cluster.h"

//test keywords
const char* keys =
"{ help h |                  | Print help message. }"
"{ tool   |      vlfeat      | Lib used for SIFT, \"opencv\" or \"vlfeat\" or \"both\", default \"vlfeat\". }"
"{ mode   |      train       | function mode, must be one of 'train', 'matching' or 'demo' }"
"{ path   |                  | Path to the image folder, set mode for different processing ways }";




/*
	Function: Test graph building function, read kecenter file from arg[1] and build graphs by our algorithm from the provided img (arg[2])
	arg[1]: path to kcenter file
	arg[2]: path to the img for graph building
*/
int graphbuildTest(int argc, const char* argv[]) {
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

/*
	Functions: test the function that read parameters from .json file
*/
void readUserTest() {
	std::string usersetting = "D:\\thesis\\Visual-Coarse-Relocalization-for-AR-Navigation-App\\User\\vrn_set.json";
	fs::path userset(usersetting);
	fileManager::read_user_set(userset);
	/*std::string dumprel = testList.dump();
	std::cout << dumprel << std::endl;

	std::cout << "read the actual params"<<fileManager::parameters::maxNumDeg << std::endl << fileManager::parameters::numOfAttemp << std::endl;*/

}

/*
	Function: verify the implementation of the Robust kernel matrix by manually build simple graphs and compute the Robust Kernel value
*/
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


/*
*	Function:
		read pretrained kcenter file as dictionary, and read two images, extracting features from img1 (arg2) and img2 (arg3), labeling features 
		with dictionary then build graphs by our algorithm and compute the Robust Kernel Matrix for the comparison of these two imgs
*	arg1: path to the kcenter file, stored as yml
*	arg2: path to image1 that does the image matching
*	arg3: path to image2 that does the image matching
*	kptKeeper: the percentage of keypoints keep for comparison
*/
void graphBuildPlusKernelTest(int argc, const char* argv[], double kptKeeper = 1.0, int iterations=1) {
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
	/*extractor::openCVimg_descips_compute(trainPath, descripts1, kpts1);*/
	//do kd-tree on the source descriptors
	std::vector<cv::DMatch> matches = matcher::kdTree(kCenters, descripts1);
	igraph_t mygraph1;
	bool status = graph::build(matches, kpts1, mygraph1);
	if (!status) { std::cout << "graph build failed! check your function." << std::endl;}

	//use the filename as graph name
	std::string graphName1 = fs::path(testImg1).stem().string();
	fileManager::write_graph(mygraph1, graphName1, "graphml");
	trainPath.pop_back();


	//read and build graph for the second test image
	//do matching on the two test image 
	std::string testImg2(argv[3]);
	trainPath.push_back(testImg2);
	cv::Mat descripts2;
	std::vector<cv::KeyPoint> kpts2;
	extractor::vlimg_descips_compute(trainPath, descripts2, kpts2);
	//keep the percentage of kpts
	std::vector<size_t> indKpts(descripts2.rows);
	std::iota(std::begin(indKpts), std::end(indKpts), 0);

	//set seed
	std::vector<uint32_t> random_data(624);
	std::random_device source;
	std::generate(random_data.begin(), random_data.end(), std::ref(source));
	std::seed_seq seeds(random_data.begin(), random_data.end());
	std::mt19937 engine(seeds);
	bool writegraph = false;
	//compute several random seeds
	for (int i = 0; i < iterations; i++) {
		cv::Mat reserveDescripts;
		std::vector<size_t> reserveKpts;
		std::vector<cv::KeyPoint> reserveKpts2;
		if (kptKeeper - 1.0 <= std::numeric_limits<double>::epsilon()) {
			reserveDescripts = descripts2;
			reserveKpts2 = kpts2;
		}
		else
		{
			std::sample(indKpts.begin(), indKpts.end(), std::back_inserter(reserveKpts), size_t(descripts2.rows * kptKeeper), engine);
			for (size_t i = 0; i < reserveKpts.size(); i++) {
				reserveDescripts.push_back(descripts2.row(reserveKpts[i]));
				reserveKpts2.push_back(kpts2[reserveKpts[i]]);
			}
		}
		
		
		std::cout << "feature keeps percents: " << double(reserveDescripts.rows) / descripts2.rows << std::endl;
		//sample for the 
		/*extractor::openCVimg_descips_compute(trainPath, descripts2, kpts2);*/
		//do kd-tree on the source descriptors
		matches = matcher::kdTree(kCenters, reserveDescripts);
		igraph_t mygraph2;
		status = graph::build(matches, reserveKpts2, mygraph2);
		if (!status) { std::cout << "graph build failed! check your function." << std::endl; }
		std::string graphName2 = fs::path(testImg2).stem().string();
		if (!writegraph) {
			fileManager::write_graph(mygraph2, graphName2, "graphml");
			writegraph = true;
		}
		
		auto sTime = clock();

		//add to the graph bags and compute the matching score
		kernel::robustKernel kernelObj(1, kCenters.rows);
		kernelObj.push_back(mygraph2);
		kernelObj.push_back(mygraph1);

		std::cout << " -> igraph spends times" << (clock() - sTime) / double(CLOCKS_PER_SEC) << " secs......" << std::endl;
		auto resMat = kernelObj.robustKernelCom();
		int cols = igraph_matrix_ncol(&resMat);
		int rows = igraph_matrix_nrow(&resMat);

		std::cout << i << "th iteration with random seeds" << std::endl;
		for (int i = 0; i < cols; i++) {
			std::cout << std::endl;
			for (int j = 0; j < rows; j++) {
				std::cout << MATRIX(resMat, i, j) << "\t";
			}
		}
		std::cout << std::endl;		
	}

}

/*
	Function: 
		read training files from arg path (argparser relies on FuntestVRN), testing the Robust kernel value under different kcenter centroid value
	args: see FunTestVRN
*/
int dictTest(int argc, const char* argv[]) {

	std::filesystem::path user_set("D:\\thesis\\Visual-Coarse-Relocalization-for-AR-Navigation-App\\User\\vrn_set.json");
	std::ofstream CSVOutput;
	CSVOutput.open("graph_comparison.csv", std::fstream::out | std::fstream::app);
	CSVOutput << "centerNo" << "," << "similarScore" << "\n";
	fileManager::read_user_set(user_set);
	std::vector<std::string> trainPaths, testPaths;
	fileManager::ArgList readResult;
	cv::Mat allDescrips, kCenters;
	std::vector<cv::KeyPoint> keypoints;
	try
	{
		readResult = fileManager::funTestRead(argc, argv, trainPaths, testPaths, keys);
	}
	catch (const std::invalid_argument& msg)
	{
		std::cout << "Exception:" << msg.what() << std::endl;
		return 0;
	}
	std::cout << "->total files found: " << trainPaths.size() + testPaths.size() << std::endl;

	std::cout<<"!------- Start feature detection with OpenCV and VLFeat ------!\n";
	if (readResult.tool == fileManager::ArgType::TOOL_VLFEAT || readResult.tool == fileManager::ArgType::TOOL_OPENCV_AND_VLFEAT) {
		std::cout<<"!------- VLFeat ------!\n";
		clock_t sTime = clock();
		//start vlfeat sift feature detection
		try {
			extractor::vlimg_descips_compute(trainPaths, allDescrips, keypoints);
		}
		catch (std::invalid_argument& e) {
			std::cout << e.what() << std::endl;
			return 0;
		};		
		//demo
		std::cout << "-> vlfeat SIFT detection / Kmeans learning totally spent " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " sec......" << std::endl;
	}
	//openCV pipeline
	if (readResult.tool == fileManager::ArgType::TOOL_OPENCV_AND_VLFEAT || readResult.tool == fileManager::ArgType::TOOL_OPENCV) {
		std::cout<<"!------- OpenCV ------!\n";
		clock_t sTime = clock();
		try {
			extractor::openCVimg_descips_compute(trainPaths, allDescrips, keypoints);
		}
		catch (std::invalid_argument& e) {
			std::cout << e.what() << std::endl;
			return 0;
		}
	
	}
	
	//test different kcenters num
	std::vector<int> centerNums = {100,150,200,250};
	clock_t sTime = clock();
	for (auto i : centerNums) {
		fileManager::parameters::centers = i;
		if (readResult.tool == fileManager::ArgType::TOOL_VLFEAT && readResult.mode == fileManager::ArgType::MODE_TRAIN) {
			//train k-means classifier
			//free memory since keypoints during training is not useful 
			std::vector <cv::KeyPoint >().swap(keypoints);
			cluster::vl_visual_word_compute(allDescrips, kCenters);
		}
		//kmeans visual word computing by openCV
		if (readResult.tool == fileManager::ArgType::TOOL_OPENCV && readResult.mode == fileManager::ArgType::MODE_TRAIN) {
			std::vector<cv::KeyPoint>().swap(keypoints);
			cluster::openCV_visual_words_compute(allDescrips, kCenters);
		}
		std::cout << "-> opencv SIFT detection / Kmeans learning totally spent " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " sec......" << std::endl;
		//write important data to file


		if (readResult.mode == fileManager::ArgType::DEFAULT || readResult.tool == fileManager::ArgType::DEFAULT) {
			std::cout << "ERROR: unsupported arguments list" << std::endl;
		}

		//use kcenters for dictionary building
		//do matching on the two test image 
		std::string testImg1 = "D:\\thesis\\Visual-Coarse-Relocalization-for-AR-Navigation-App\\imgs\\simple\\test\\IMG_20200527_145625.jpg";
		std::string testImg2 = "D:\\thesis\\Visual-Coarse-Relocalization-for-AR-Navigation-App\\imgs\\simple\\test\\IMG_20200527_145701.jpg";

		std::vector<std::string> testImgPath;
		testImgPath.push_back(testImg1);
		cv::Mat descripts1;
		std::vector<cv::KeyPoint> kpts1;

		if (readResult.tool == fileManager::ArgType::TOOL_OPENCV) {
			extractor::openCVimg_descips_compute(testImgPath, descripts1, kpts1);
		}
		else {
			extractor::vlimg_descips_compute(testImgPath, descripts1, kpts1);
		}
		//do kd-tree on the source descriptors
		std::vector<cv::DMatch> matches = matcher::kdTree(kCenters, descripts1);
		igraph_t mygraph1;
		bool status = graph::build(matches, kpts1, mygraph1);
		if (!status) { std::cout << "graph build failed! check your function." << std::endl; }

		fileManager::write_graph(mygraph1, "kcentertestimg1", "graphml");
		testImgPath.pop_back();


		//read and build graph for the second test image
		//do matching on the two test image 
		testImgPath.push_back(testImg2);
		cv::Mat descripts2;
		std::vector<cv::KeyPoint> kpts2;

		if (readResult.tool == fileManager::ArgType::TOOL_OPENCV) {
			extractor::openCVimg_descips_compute(testImgPath, descripts2, kpts2);
		}
		else {
			extractor::vlimg_descips_compute(testImgPath, descripts2, kpts2);
		}

		//do kd-tree on the source descriptors
		matches = matcher::kdTree(kCenters, descripts2);
		igraph_t mygraph2;
		status = graph::build(matches, kpts2, mygraph2);
		if (!status) { std::cout << "graph build failed! check your function." << std::endl; }
		fileManager::write_graph(mygraph2, "kcentertestimg2", "graphml");

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
		CSVOutput << i << "," << MATRIX(resMat, 0, 1) << "\n";
	}

}
//int main(int argc, const char* argv[]) {
//	igraph_i_set_attribute_table(&igraph_cattribute_table);
//	readUserTest();
//	double keeps = 1.0;
//	graphBuildPlusKernelTest(argc, argv, keeps, 1);
//	/*dictTest(argc, argv);*/
//}



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