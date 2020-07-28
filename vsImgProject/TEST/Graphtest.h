#ifndef _GRAPHTEST_H
#define _GRAPHTEST_H

#include "StaticVRImg/graph.h"
#include "StaticVRImg/extractor.h"
#include "StaticVRImg/fileManager.h"
#include "StaticVRImg/matcher.h"
#include "StaticVRImg/kernel.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
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
inline const char* keys =
"{ help h |                  | Print help message. }"
"{ tool   |      vlfeat      | Lib used for SIFT, \"opencv\" or \"vlfeat\" or \"both\", default \"vlfeat\". }"
"{ mode   |      train       | function mode, must be one of 'train', 'matching' or 'demo' }"
"{ path   |                  | Path to the image folder, set mode for different processing ways }";

inline void covisMapTest() {
	//define testing graphs
	auto sTime = clock();
	igraph_t testGraph;
	igraph_i_set_attribute_table(&igraph_cattribute_table);
	igraph_real_t posx1[] = { 1.0,1.0,3.0,5.0,5.0,7.0,7,7 };
	igraph_real_t posy1[] = { 1.0,3.0,2.0,1.0,3.0,2.0,3,1 };
	igraph_real_t edges[] = { 0,1,0,2,1,2,2,3,2,4,3,4,3,5,4,5,4,6,3,7 };
	igraph_real_t label1[] = { 0,1,2,3,4,5,6,7 };
	igraph_real_t scale1[] = { 1,1,1,1,1,1,1,1 };
	igraph_real_t edgeW1[] = { 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1,1 };
	igraph_vector_t e1, px1, py1, lab1, scl1, ew1;

	igraph_vector_view(&e1, edges, sizeof(edges) / sizeof(double));
	igraph_vector_view(&px1, posx1, sizeof(posx1) / sizeof(double));
	igraph_vector_view(&py1, posy1, sizeof(posy1) / sizeof(double));
	igraph_vector_view(&lab1, label1, sizeof(label1) / sizeof(double));
	igraph_vector_view(&scl1, scale1, sizeof(scale1) / sizeof(double));
	igraph_vector_view(&ew1, edgeW1, sizeof(edgeW1) / sizeof(double));
	igraph_create(&testGraph, &e1, 0, IGRAPH_UNDIRECTED);
	SETGAS(&testGraph, "name", "testGraph1");
	SETGAN(&testGraph, "n_vertices", sizeof(scale1) / sizeof(double));
	SETEANV(&testGraph, "weight", &ew1);
	SETVANV(&testGraph, "label", &lab1);
	SETVANV(&testGraph, "posx", &px1);
	SETVANV(&testGraph, "posy", &py1);
	SETVANV(&testGraph, "scale", &scl1);

	//create testgraph2
	igraph_t testGraph2;
	igraph_vector_t e2, px2, py2, lab2, scl2, ew2;
	igraph_real_t posx2[] = { 3,5,4,3,5,4,3,5 };
	igraph_real_t posy2[] = { 0,0,2,3,3,4,2,2 };
	igraph_real_t edges2[] = { 0,1,0,2,1,2,2,3,2,4,3,4,3,5,4,5,0,6,1,7 };
	igraph_real_t label2[] = { 0,1,2,8,8,8,6,7 };
	igraph_real_t scale2[] = { 1,1,1,1,1,1,1,1 };
	igraph_real_t edgeW2[] = { 1,1,1,1,1,1,1,1,1,1 };

	igraph_vector_view(&e2, edges2, sizeof(edges2) / sizeof(double));
	igraph_vector_view(&px2, posx2, sizeof(posx2) / sizeof(double));
	igraph_vector_view(&py2, posy2, sizeof(posy2) / sizeof(double));
	igraph_vector_view(&lab2, label2, sizeof(label2) / sizeof(double));
	igraph_vector_view(&scl2, scale2, sizeof(scale2) / sizeof(double));
	igraph_vector_view(&ew2, edgeW2, sizeof(edgeW2) / sizeof(double));
	igraph_create(&testGraph2, &e2, 0, IGRAPH_UNDIRECTED);
	SETGAS(&testGraph2, "name", "testGraph2");
	SETGAN(&testGraph2, "n_vertices", sizeof(scale2) / sizeof(double));
	SETEANV(&testGraph2, "weight", &ew2);
	SETVANV(&testGraph2, "label", &lab2);
	SETVANV(&testGraph2, "posx", &px2);
	SETVANV(&testGraph2, "posy", &py2);
	SETVANV(&testGraph2, "scale", &scl2);
	
	//testGraph3
	igraph_t testGraph3;
	igraph_vector_t e3, px3, py3, lab3, scl3, ew3;
	igraph_real_t posx3[] = { 1,1,1,3,3,5,7,7,9,9,9 };
	igraph_real_t posy3[] = { 1,2,3,3,1,2,1,3,2,3,1 };
	igraph_real_t edges3[] = { 0,1,0,4,1,2,2,3,3,4,3,5,4,5,5,6,5,7,6,7,6,8,6,10,7,8,7,9};
	igraph_real_t label3[] = { 9,9,9,1,0,2,3,4,5,6,7};
	igraph_real_t scale3[] = { 1,1,1,1,1,1,1,1,1,1,1 };
	igraph_real_t edgeW3[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1 };

	igraph_vector_view(&e3, edges3, sizeof(edges3) / sizeof(double));
	igraph_vector_view(&px3, posx3, sizeof(posx3) / sizeof(double));
	igraph_vector_view(&py3, posy3, sizeof(posy3) / sizeof(double));
	igraph_vector_view(&lab3, label3, sizeof(label3) / sizeof(double));
	igraph_vector_view(&scl3, scale3, sizeof(scale3) / sizeof(double));
	igraph_vector_view(&ew3, edgeW3, sizeof(edgeW3) / sizeof(double));
	igraph_create(&testGraph3, &e3, 0, IGRAPH_UNDIRECTED);
	SETGAS(&testGraph3, "name", "testGraph3");
	SETGAN(&testGraph3, "n_vertices", sizeof(scale3) / sizeof(double));
	SETEANV(&testGraph3, "weight", &ew3);
	SETVANV(&testGraph3, "label", &lab3);
	SETVANV(&testGraph3, "posx", &px3);
	SETVANV(&testGraph3, "posy", &py3);
	SETVANV(&testGraph3, "scale", &scl3);

	//test the graph extension function
	std::vector<cv::DMatch> bestMatches12,bestMatches23,bestMatches13;
	bestMatches12.push_back(cv::DMatch(0, 4, 0));
	bestMatches12.push_back(cv::DMatch(1, 3, 0));
	bestMatches12.push_back(cv::DMatch(2, 5, 0));
	bestMatches12.push_back(cv::DMatch(6, 6, 0));
	bestMatches12.push_back(cv::DMatch(7, 7, 0));

	bestMatches13.push_back(cv::DMatch(3, 1, 0));
	bestMatches13.push_back(cv::DMatch(4, 0, 0));
	bestMatches13.push_back(cv::DMatch(5, 2, 0));
	bestMatches13.push_back(cv::DMatch(6, 3, 0));
	bestMatches13.push_back(cv::DMatch(7, 4, 0));
	bestMatches13.push_back(cv::DMatch(8, 5, 0));
	bestMatches13.push_back(cv::DMatch(9, 6, 0));
	bestMatches13.push_back(cv::DMatch(10, 7, 0));

	bestMatches23.push_back(cv::DMatch(6, 1, 0));
	bestMatches23.push_back(cv::DMatch(7, 0, 0));
	bestMatches23.push_back(cv::DMatch(8, 2, 0));
	bestMatches23.push_back(cv::DMatch(9, 6, 0));
	bestMatches23.push_back(cv::DMatch(10, 7, 0));

	fileManager::write_graph(testGraph, "covisTestG1", "graphml");
	fileManager::write_graph(testGraph2, "covisTestG2", "graphml");
	fileManager::write_graph(testGraph3, "covisTestG3", "graphml");

	std::vector<igraph_t*> graphs;
	kernel::covisMap covis(10);
	covis.process(testGraph);
	graphs.push_back(&testGraph);
	covis.process(testGraph2);
	graphs.push_back(&testGraph2);
	covis.process(testGraph3);
	graphs.push_back(&testGraph3);
	/*covis.printMap();*/

	auto rel = covis.retrieve(testGraph);
	for (int i = 0; i < rel.size(); i++) {
		for (int j = 0; j < rel[i].size(); j++) {
			std::cout << j << " ";
		}
		std::cout << "\n";
	}

	//build virtual locations, single graph location is neglected
	igraph_vector_t G123posx, G123posy;
	igraph_vector_init(&G123posx, 0);
	igraph_vector_init(&G123posy, 0);
	graph::extend(testGraph, testGraph2, bestMatches12);
	fileManager::write_graph(testGraph, "covisTestG12", "graphml");
	graph::extend(testGraph, testGraph3,bestMatches13);
	fileManager::write_graph(testGraph, "covisTestG123", "graphml");
	graph::extend(testGraph2, testGraph3, bestMatches23);
	fileManager::write_graph(testGraph2, "covisTestG23", "graphml");
}
inline void graphExtendTest() {
	//define testing graphs
	auto sTime = clock();
	igraph_t testGraph;
	igraph_i_set_attribute_table(&igraph_cattribute_table);
	igraph_real_t posx1[] = {1.0,1.0,3.0,5.0,5.0,7.0,7,7};
	igraph_real_t posy1[] = { 1.0,3.0,2.0,1.0,3.0,2.0,3,1 };
	igraph_real_t edges[] = { 0,1,0,2,1,2,2,3,2,4,3,4,3,5,4,5,4,6,3,7 };
	igraph_real_t label1[] = { 0,1,2,3,4,5,6,7 };
	igraph_real_t scale1[] = { 1,1,1,1,1,1,1,1 };
	igraph_real_t edgeW1[] = { 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1,1 };
	igraph_vector_t e1, px1, py1, lab1,scl1,ew1;
	
	igraph_vector_view(&e1, edges, sizeof(edges) / sizeof(double));
	igraph_vector_view(&px1, posx1, sizeof(posx1) / sizeof(double));
	igraph_vector_view(&py1, posy1, sizeof(posy1) / sizeof(double));
	igraph_vector_view(&lab1, label1, sizeof(label1) / sizeof(double));
	igraph_vector_view(&scl1, scale1, sizeof(scale1) / sizeof(double));
	igraph_vector_view(&ew1, edgeW1, sizeof(edgeW1) / sizeof(double));
	igraph_create(&testGraph, &e1, 0, IGRAPH_UNDIRECTED);
	SETGAS(&testGraph, "name", "testGraph1");
	SETGAN(&testGraph, "n_vertices", sizeof(scale1) / sizeof(double));
	SETEANV(&testGraph, "weight", &ew1);
	SETVANV(&testGraph, "label", &lab1);
	SETVANV(&testGraph, "posx", &px1);
	SETVANV(&testGraph, "posy", &py1);
	SETVANV(&testGraph, "scale", &scl1);
	
	//create testgraph2
	igraph_t testGraph2;
	igraph_vector_t e2, px2, py2, lab2, scl2, ew2;
	igraph_real_t posx2[] = {3,5,4,3,5,4,3,5};
	igraph_real_t posy2[] = { 0,0,2,3,3,4,2,2 };
	igraph_real_t edges2[] = { 0,1,0,2,1,2,2,3,2,4,3,4,3,5,4,5,0,6,1,7 };
	igraph_real_t label2[] = { 0,1,2,3,4,5,6,7 };
	igraph_real_t scale2[] = { 1,1,1,1,1,1,1,1 };
	igraph_real_t edgeW2[] = { 1,1,1,1,1,1,1,1,1,1 };

	igraph_vector_view(&e2, edges2, sizeof(edges2) / sizeof(double));
	igraph_vector_view(&px2, posx2, sizeof(posx2) / sizeof(double));
	igraph_vector_view(&py2, posy2, sizeof(posy2) / sizeof(double));
	igraph_vector_view(&lab2, label2, sizeof(label2) / sizeof(double));
	igraph_vector_view(&scl2, scale2, sizeof(scale2) / sizeof(double));
	igraph_vector_view(&ew2, edgeW2, sizeof(edgeW2) / sizeof(double));
	igraph_create(&testGraph2, &e2, 0, IGRAPH_UNDIRECTED);
	SETGAS(&testGraph2, "name", "testGraph2");
	SETGAN(&testGraph2, "n_vertices", sizeof(scale2) / sizeof(double));
	SETEANV(&testGraph2, "weight", &ew2);
	SETVANV(&testGraph2, "label", &lab2);
	SETVANV(&testGraph2, "posx", &px2);
	SETVANV(&testGraph2, "posy", &py2);
	SETVANV(&testGraph2, "scale", &scl2);
	
	//test the graph extension function
	std::vector<cv::DMatch> bestMatches;
	bestMatches.push_back(cv::DMatch(0,4,0));
	bestMatches.push_back(cv::DMatch(1, 3, 0));
	bestMatches.push_back(cv::DMatch(2, 5, 0));
	bestMatches.push_back(cv::DMatch(6, 6, 0));
	bestMatches.push_back(cv::DMatch(7, 7, 0));
	fileManager::write_graph(testGraph, "testG1", "graphml");
	fileManager::write_graph(testGraph2, "testG2", "graphml");
	graph::extend(testGraph, testGraph2, bestMatches);

	//write the testGraph
	fileManager::write_graph(testGraph, "extendtest", "graphml");
}



/*
	Function: Test graph building function, read kecenter file from arg[1] and build graphs by our algorithm from the provided img (arg[2])
	arg[1]: path to kcenter file
	arg[2]: path to the img for graph building
	arg[3]: path to the second img path for graph building
*/
inline int graphbuildTest(int argc, const char* argv[]) {
	//testing the build function with graph
	if (argc < 4) {
		std::cout << "Please provides path to the dictionary and two images!" << std::endl;
	}
	cv::FileStorage reader;
	reader.open(argv[1],cv::FileStorage::READ);
	if (!reader.isOpened()) { std::cout << "failed to open the kcenter file" << std::endl; }
	//read kcenters
	cv::Mat kCenters;
	reader["kcenters"] >> kCenters;

	//do detection on the test image 
	std::string testImg1(argv[2]);
	cv::Mat img1 = cv::imread(testImg1, cv::IMREAD_GRAYSCALE);
	cv::Mat descripts1;
	std::vector<cv::KeyPoint> kpts1;
	extractor::vlimg_descips_compute_simple(img1, descripts1, kpts1);

	std::string testImg2(argv[3]);
	cv::Mat img2 = cv::imread(testImg2, cv::IMREAD_GRAYSCALE);
	cv::Mat descripts2;
	std::vector<cv::KeyPoint> kpts2;
	extractor::vlimg_descips_compute_simple(img2, descripts2, kpts2);

	//do matching and choose the best match
	matcher::kdTree matches(descripts1);
	auto bestMatches = matches.search(descripts2);

	//records the matching keypoints by the matching result
	std::vector<cv::Point2f> matchkpts1;
	std::vector<cv::Point2f> matchkpts2;
	for (int i = 0; i < bestMatches.size(); i++) {
		matchkpts1.push_back(kpts1[bestMatches[i].trainIdx].pt);
		matchkpts2.push_back(kpts2[bestMatches[i].queryIdx].pt);
	}
	cv::Mat mask;
	cv::Mat homo = cv::findHomography(matchkpts1, matchkpts2,mask,cv::RANSAC);

	//print out the homograph mat
	int maskNum=0;
	std::cout << "transformation matrix is: " << homo << std::endl;

	//according to the mask to merge points
	std::vector<cv::Point2f> outliers;

	for (int i = 0; i < mask.rows; i++) {
		if (!mask.at<uchar>(i)) {
			//add outliers to the keypoints and descripts
			outliers.push_back(matchkpts2[i]);
			descripts1.push_back(descripts2.row(bestMatches[i].queryIdx));
		}
	}
	std::vector<cv::Point2f> outliers_trans(outliers.size());

	//compute the outliers' transformation
	cv::perspectiveTransform(outliers, outliers_trans, homo);

	std::vector<cv::KeyPoint> kpts2Trans;
	for (int i = 0; i < mask.rows; i++) {
		if (!mask.at<uchar>(i)) {
			//add outliers to the keypoints and descripts
			kpts2[bestMatches[i].queryIdx].pt= outliers_trans[i];
			kpts1.push_back(kpts2[bestMatches[i].queryIdx]);
		}
	}

	//do kd-tree on the source descriptors
	matcher::kdTree kdtreeMatcher(kCenters);
	std::vector<cv::DMatch> matchesKcenter = kdtreeMatcher.search(descripts1);

	igraph_t mygraph;
	bool status = graph::build(matchesKcenter, kpts1, mygraph);
	if (!status) { std::cout << "graph build failed! check your function." << std::endl; return 0; }

	fileManager::write_graph(mygraph, "GraphbuildRANSAC", "graphml");
}

/*
	Functions: test the function that read parameters from .json file
*/
inline void readUserTest() {
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
inline void graphKernelTest() {
	
	auto sTime = clock();
	igraph_t testGraph;
	igraph_vector_t e;
	igraph_real_t edges[] = { 0,1,0,8,0,9,1,8,1,2,2,3,2,7,3,4,3,6,4,5,4,6,5,6,6,12,7,8,7,11,8,10,9,10,10,11,11,12};
	igraph_vector_view(&e, edges, sizeof(edges) / sizeof(double));
	igraph_i_set_attribute_table(&igraph_cattribute_table);
	igraph_create(&testGraph, &e, 0, IGRAPH_UNDIRECTED);

	//create testgraph2
	igraph_t testGraph2;
	igraph_vector_t e2;
	igraph_real_t edges2[] = { 0,1,0,5,1,2,1,4,3,4,4,6,5,6,6,7 };
	igraph_vector_view(&e2, edges2, sizeof(edges2) / sizeof(double));
	igraph_create(&testGraph2, &e2, 0, IGRAPH_UNDIRECTED);
	//add labels
	std::vector<int> labstd{ 0,2,1,4,0,3,0,3,1,4,0,1,2 };
	std::vector<double> labstd_vec(labstd.begin(), labstd.end());
	igraph_vector_t lab_vec,lab_vec2;
	igraph_real_t labs[] = { 0,2,1,4,0,3,0,3,1,4,0,1,2 };
	
	igraph_vector_view(&lab_vec, labs, sizeof(labs)/sizeof(double));
	
	igraph_real_t labs2[] = { 0,2,1,3,1,4,0,1 };
	igraph_vector_view(&lab_vec2, labs2, sizeof(labs2) / sizeof(double));
	SETVANV(&testGraph, "label", &lab_vec);
	SETVANV(&testGraph2, "label", &lab_vec2);

	//defines the kernel class
	kernel::robustKernel kernelObj(1, 5);
	kernelObj.push_back(testGraph);
	std::vector<igraph_t> queryGraph;
	queryGraph.push_back(testGraph2);

	auto scores = kernelObj.robustKernelCompWithQueryArray(queryGraph);
	kernelObj.push_back(testGraph2);
	auto k_matrix = kernelObj.robustKernelCom();
	std::cout << "scores :" << scores[0][0]<<std::endl;

	//manually create the inverted_index vectors
	std::vector<std::unordered_map<int, std::vector<size_t>>> inverts;
	std::unordered_map<int, std::vector<size_t>> invert1{ {0,std::vector<size_t>{0,4,6,10}},
		{1,std::vector<size_t>{2,8,11}},{2,std::vector<size_t>{1,12}},{3,std::vector<size_t>{5,7}},{4,std::vector<size_t>{3,9}} };
	std::unordered_map<int, std::vector<size_t>> invert2{ {0,std::vector<size_t>{0,6}},
		{1,std::vector<size_t>{2,4,7}},{2,std::vector<size_t>{1}},{3,std::vector<size_t>{3}},{4,std::vector<size_t>{5}} };
	inverts.push_back(invert1);
	inverts.push_back(invert2);
	double values=0.0;
	for (int i = 0; i < invert1.size();i++) {
		values+=kernelObj.robustKernelVal(invert1[i], invert2[i], testGraph, testGraph2);
	}

	std::cout << "kernel values: " << values * (1.0 / (8+19)) * (1.0 / (8+19)) <<std::endl;
	std::cout << "kernel class computed value: \n";
	//manully compute the kernel value
	std::cout << " -> igraph spends times" << (clock() - sTime) / double(CLOCKS_PER_SEC) << " secs......" << std::endl;

	int cols = igraph_matrix_ncol(&k_matrix);
	int rows = igraph_matrix_nrow(&k_matrix);
	for (int i = 0; i < cols; i++) {
		std::cout << std::endl;
		for (int j = 0; j < rows; j++) {
			std::cout << MATRIX(k_matrix, i, j)<<"\t";
		}
	}
	//pure kernel value computation
	
	
	//test with old functions
	/*sTime = clock();
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
	std::cout << std::endl<<k_mat;*/
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
inline void graphBuildPlusKernelTest(int argc, const char* argv[], double kptKeeper = 1.0, int iterations=1) {
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
	cv::Mat descripts1;
	std::vector<cv::KeyPoint> kpts1;

	if (!fs::exists(fs::path(testImg1))) {
		std::cout << "vlfeat sift feature detection: Warning: " << testImg1 << " does not exist!" << std::endl;
		return;
	}
	cv::Mat grayImg;
	cv::cvtColor(cv::imread(testImg1), grayImg, cv::COLOR_BGR2GRAY);


	extractor::vlimg_descips_compute_simple(grayImg, descripts1, kpts1);
	/*extractor::openCVimg_descips_compute(trainPath, descripts1, kpts1);*/
	//do kd-tree on the source descriptors
	matcher::kdTree kdtreeMatcher(kCenters);
	std::vector<cv::DMatch> matches = kdtreeMatcher.search(descripts1);
	igraph_t mygraph1;
	bool status = graph::build(matches, kpts1, mygraph1);
	if (!status) { std::cout << "graph build failed! check your function." << std::endl;}

	//use the filename as graph name
	std::string graphName1 = fs::path(testImg1).stem().string();
	fileManager::write_graph(mygraph1, graphName1, "graphml");

	//read and build graph for the second test image
	//do matching on the two test image 
	std::string testImg2(argv[3]);
	cv::Mat descripts2;
	std::vector<cv::KeyPoint> kpts2;

	if (!fs::exists(fs::path(testImg2))) {
		std::cout << "vlfeat sift feature detection: Warning: " << testImg2 << " does not exist!" << std::endl;
		return;
	}
	cv::Mat grayImg2;
	cv::cvtColor(cv::imread(testImg2), grayImg2, cv::COLOR_BGR2GRAY);
	extractor::vlimg_descips_compute_simple(grayImg2, descripts2, kpts2);
	//keep the percentage of kpts
	std::vector<size_t> indKpts(descripts2.rows);
	std::iota(std::begin(indKpts), std::end(indKpts), 0);

	//set seed
	/*std::vector<uint32_t> random_data(624);
	std::random_device source;
	std::generate(random_data.begin(), random_data.end(), std::ref(source));
	std::seed_seq seeds(random_data.begin(), random_data.end());
	std::mt19937 engine(seeds);*/
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
			std::sample(indKpts.begin(), indKpts.end(), std::back_inserter(reserveKpts), size_t(descripts2.rows * kptKeeper), std::mt19937{ std::random_device{}() });
			for (size_t i = 0; i < reserveKpts.size(); i++) {
				reserveDescripts.push_back(descripts2.row(reserveKpts[i]));
				reserveKpts2.push_back(kpts2[reserveKpts[i]]);
			}
		}
		
		std::cout << "feature keeps percents: " << double(reserveDescripts.rows) / descripts2.rows << std::endl;
		//sample for the 
		/*extractor::openCVimg_descips_compute(trainPath, descripts2, kpts2);*/
		//do kd-tree on the source descriptors
		matches = kdtreeMatcher.search(reserveDescripts);

		igraph_t mygraph2;
		status = graph::build(matches, reserveKpts2, mygraph2);
		if (!status) { std::cout << "graph build failed! check your function." << std::endl; }
		std::string graphName2 = fs::path(testImg2).stem().string();
		if (!writegraph) {
			fileManager::write_graph(mygraph2, graphName2, "graphml");
			writegraph = true;
		}
		
		auto sTime = clock();

		//	//add to the graph bags and compute the matching score
		kernel::robustKernel kernelObj(1, kCenters.rows);
		kernelObj.push_back(mygraph2);
		/*kernelObj.push_back(mygraph1);*/

		std::vector<igraph_t>query_test;
		query_test.push_back(mygraph1);
		std::cout << " -> igraph spends times" << (clock() - sTime) / double(CLOCKS_PER_SEC) << " secs......" << std::endl;

		auto resMat = kernelObj.robustKernelCompWithQueryArray(query_test);
		/*int cols = igraph_matrix_ncol(&resMat);
		int rows = igraph_matrix_nrow(&resMat);
		igraph_destroy(&mygraph2);
		std::cout << i << "th iteration with random seeds" << std::endl;
		for (int i = 0; i < cols; i++) {
				std::cout << std::endl;
				for (int j = 0; j < rows; j++) {
					std::cout << MATRIX(resMat, i, j) << "\t";
				}
		}*/
		for (int i = 0; i < resMat.size(); i++) {
			std::cout << std::endl;
			for (int j = 0; j < resMat[i].size(); j++) {
				std::cout << resMat[i][j] << "\t";
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
inline int dictTest(int argc, const char* argv[]) {

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
	matcher::kdTree kdtreeMatcher(kCenters);

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
		std::vector<cv::DMatch> matches = kdtreeMatcher.search(descripts1);
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
		matches = kdtreeMatcher.search(descripts2);
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


#endif // !_GRAPHTEST_H
//int main(int argc, const char* argv[]) {
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