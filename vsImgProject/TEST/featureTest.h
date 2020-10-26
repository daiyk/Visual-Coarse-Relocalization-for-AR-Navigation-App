#pragma once
#ifndef _FEATURETEST_H
#define _FEATURETEST_H
#include "nbhdGraph.h"
#include <fstream>
#include <StaticVRImg/fileManager.h>
#include <StaticVRImg/extractor.h>
#include <StaticVRImg/matcher.h>
#include <StaticVRImg/FLANN.h>
#include <StaticVRImg/graph.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <unordered_map>
#include <colmap/util/endian.h>
#include <colmap/base/database.h>
#include <colmap/feature/sift.h>


using params = fileManager::parameters;
template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
void eigen2cv(const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& src, cv::Mat& dst)
{
	if (!(src.Flags & Eigen::RowMajorBit))
	{
		cv::Mat _src(src.cols(), src.rows(), cv::DataType<_Tp>::type,
			(void*)src.data(), src.stride() * sizeof(_Tp));
		cv::transpose(_src, dst);
	}
	else
	{
		cv::Mat _src(src.rows(), src.cols(), cv::DataType<_Tp>::type,
			(void*)src.data(), src.stride() * sizeof(_Tp));
		_src.copyTo(dst);
	}
}
//void featureExtTest(){
//	std::string imgPath = "E:\\datasets\\south-building\\images_database\\P1180141.JPG";
//	colmap::Bitmap img;
//	img.Read(imgPath, false);
//	//test the extractor
//	std::vector<colmap::Bitmap> imgs;
//	imgs.push_back(img);
//	std::vector<colmap::FeatureKeypoints> keypoints;
//	std::vector<colmap::FeatureDescriptors> descripts;
//	extractor::siftGPU_descips_compute_simple(imgs, keypoints, descripts);
//
//
//}
inline void flanntest(std::vector<std::string> query_paths, std::string database_path) {
	std::string img_folder = fs::path(query_paths[0]).parent_path().string();
	//loop over the database path and get the path for database images
	std::vector<std::string> db_imgs;
	fileManager::read_files_in_path(database_path, db_imgs);
	if (db_imgs.empty()) {
		std::cerr << "\nflanntest: error! db images are zero.";
		return;
	}
	std::vector < cv::Mat> allDescripts;

	std::ofstream flann_writer;
	flann_writer.open("flannTestResult.csv", std::fstream::out | std::fstream::app);
	flann_writer << "qry_img[ID]" << "," << "db_img[ID]" << "," << "score" << "\n";

	//iterate through the query image paths and store for comparison
	std::vector<std::string> qry_names;
	for (int i = 0; i < query_paths.size(); i++) {
		if (!fs::exists(fs::path(query_paths[i]))) {
			std::cout << "\nflanntest: the query path " << query_paths[i] << " doesn't exist!";
			continue;
		}
		qry_names.push_back(fs::path(query_paths[i]).filename().string());
		auto grayImg = cv::imread(query_paths[i], cv::IMREAD_GRAYSCALE);
		cv::Mat descripts;
		std::vector<cv::KeyPoint> kpts;
		try {
			extractor::vlimg_descips_compute_simple(grayImg,descripts,kpts);
		}
		catch (std::invalid_argument& e) {
			std::cout << e.what() << std::endl;
			break;
		};
		allDescripts.push_back(descripts);
	}
	std::vector<std::vector<double>> scores(allDescripts.size(), std::vector<double>(db_imgs.size()));
	for (int i = 0; i < db_imgs.size(); i++) {
		auto dbImg = cv::imread(db_imgs[i], cv::IMREAD_GRAYSCALE);
		cv::Mat dbDescripts;
		std::vector < cv::KeyPoint> dbKpts;


		try {
			extractor::vlimg_descips_compute_simple(dbImg, dbDescripts, dbKpts);
		}
		catch (std::invalid_argument& e) {
			std::cout << e.what() << std::endl;
			break;
		};


		//do comparison with the qry images
		for (int k = 0; k < allDescripts.size(); k++) {
			std::vector < cv::DMatch> matches;
			dbDescripts.convertTo(dbDescripts, CV_32F);
			cv::Mat qryDescrips;
			allDescripts[k].convertTo(qryDescrips, CV_32F);
			FLANN::FLANNMatch(dbDescripts, qryDescrips, matches);
			scores[k][i] = FLANN::FLANNScore(matches);
		}
	}
	for (int i = 0; i < allDescripts.size(); i++) {
		for (int j = 0; j < db_imgs.size(); j++) {
			flann_writer << qry_names[i] << "," << j << "," << scores[i][j] << "\n";
		}
	}
	
}

inline void nhhdGraphTest(std::vector<std::string> query_paths, std::string database_path) {
	fileManager::covisOptions options;
	options.database_path = database_path;
	colmap::Database databs(options.database_path);
	options.vocab_path = "E:\\vocab_tree_flickr100K_words32K.bin";
	fs::path img(query_paths[0]);
	std::string img_folder = img.parent_path().string();
	options.image_path = img_folder;
	options.numImageToKeep = 3;
	options.image_list.clear();
	if (query_paths.empty()) { std::cerr << "\nempty query_paths!"; return; }
	for (int i = 0; i < 2; i++) {
		fs::path imgpath(query_paths[i]);
		options.image_list.push_back(imgpath.filename().string());
	}
	
	//build nbhd object
	nbhd::nbhdGraph graphObj(options);
	while (true) {
		int status = graphObj.Next();
		if (status==0) {
			std::cerr << "error happens during graphObj operations" << std::endl;
			break;
		}
		if (status==1) {
			break; //finish and break
		}
	}
}
inline void vocabReadTest(std::string &imgPath) {
	//take the first arguments as image path
	colmap::Bitmap imagereader, imagereader2;
	if (!imagereader.Read(imgPath, false)) {
		std::cout << "ERROR: read image path failed!" << std::endl;
		return;
	}
	imagereader2 = imagereader.Clone();
	colmap::SiftExtractionOptions options;
	colmap::FeatureDescriptors colmap_descriptors;
	colmap::FeatureKeypoints colmap_keypoints;
	options.max_num_features = fileManager::parameters::maxNumFeatures;
	//opencv reader images
	auto img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
	cv::Mat imgBitmap(imagereader.Height(), imagereader.Width(), CV_8U);
	auto dataFloat = imagereader2.ConvertToRowMajorArray();

	//print the image difference
	std::cout << "\nreader width" << imagereader.Width() <<"\nreader height"<< imagereader.Height() <<"\n"<<(int)dataFloat.size();
	std::cout << "\nimg width" << img.size().width << "\nimg height" << img.size().height << "\n";

	//transform bitmap to opencv image
	for (int i = 0; i < imagereader2.Height(); i++) {
		for (int j = 0; j < imagereader2.Width(); j++) {
			imgBitmap.at<uchar>(i, j) = dataFloat[j + i* imagereader2.Width()];
		}
	}
	//cv::namedWindow("opencv img ", cv::WINDOW_NORMAL);// Create a window for display.
	//
	//cv::imshow("opencv img", img);

	//cv::namedWindow("bitmap img ", cv::WINDOW_NORMAL);// Create a window for display.
	//
	//cv::imshow("bitmap img", imgBitmap);
	//
	//for (int i = 0; i < 100; i++) {
	//	std::cout << i << " th img" << (int)img.data[i]<<std::endl;
	//	std::cout << i << " th bitmap" << (int)dataFloat[i] << std::endl;
	//}
	
	/*colmap::ExtractSiftFeaturesCPU(options,imagereader,&colmap_keypoints,&colmap_descriptors);*/
	
	//get the number of keypoints
	/*std::cout << "\nnumber of keypoints extracted by colmap SIFT: " << colmap_keypoints.size();*/
	std::string databasePath = "E:\\datasets\\south-building\\southbuildingdatabase.db";
	colmap::Database* db = new colmap::Database(databasePath);
	
	
	std::string vocabPath = "E:\\vocab_tree_flickr100K_words32K.bin";
	std::ifstream file(vocabPath, std::ios::binary);
	CHECK(file.is_open()) << vocabPath;
	const uint64_t rows = colmap::ReadBinaryLittleEndian<uint64_t>(&file);
	const uint64_t cols = colmap::ReadBinaryLittleEndian<uint64_t>(&file);


	//read dict centroids
	cv::Mat dict(rows, cols, CV_8U);
	dict.reserveBuffer(cols * rows); // reserve space for 
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			dict.at<uint8_t>(i, j) = colmap::ReadBinaryLittleEndian<uint8_t>(&file);
		}
	}

	//use own sift extractor
	cv::Mat cv_descriptors;
	std::vector<cv::KeyPoint> cv_keypoints;

	auto testImg = db->ReadImage(1);
	std::cout<<"\ntest image is: "<<testImg.Name();
	colmap_descriptors = db->ReadDescriptors(1);
	/*auto kpts_ = database.ReadKeypoints(1);*/
	extractor::vlimg_descips_compute_simple(imgBitmap,cv_descriptors,cv_keypoints);
	/*std::cout << "\nown sift extractor extracts keypoints: " << cv_keypoints.size();*/
	
	//compares the descriptors: print it out
	for (int i = 0; i < 4; i++) {
		std::cout << "\ncolmap descripts: "<<colmap_descriptors.rows()<<"\n" << colmap_descriptors.row(colmap_descriptors.rows()-1-i).cast<int>();// << "\ncolmap keyPoints: " << kpts_[i].x << "  " << kpts_[i].y << "  " << std::sqrt(std::pow(kpts_[i].a11, 2)
			//+ std::pow(colmap_keypoints[i].a21, 2));
		std::cout << "\n\ncv descripts: "<< cv_descriptors.rows<<"\n" << cv_descriptors.row(cv_descriptors.rows-i-1) << "\ncv keypoints: "<<cv_keypoints[cv_keypoints.size()-1-i].pt.x<<"  "<< cv_keypoints[cv_keypoints.size() - 1 - i].pt.y<<"  "<<cv_keypoints[cv_keypoints.size() - 1 - i].size;
	}


	//print out
	//test the visual id identifier working in the right way
}
inline void databaseTest() {
	/*std::string databasePath = "E:\\datasets\\gerrard-hall\\gerrard-hall-short\\gerrard-hall-short.db";*/
	std::string databasePath = "E:\\datasets\\south-building\\southbuildingDB.db";
	std::string vocabPath = "E:\\vocab_tree_flickr100K_words32K.bin";
	std::ifstream file(vocabPath, std::ios::binary);
	CHECK(file.is_open()) << vocabPath;
	const uint64_t rows = colmap::ReadBinaryLittleEndian<uint64_t>(&file);
	const uint64_t cols = colmap::ReadBinaryLittleEndian<uint64_t>(&file);

	//read descriptors
	cv::Mat dict(rows, cols, CV_8U);
	dict.reserveBuffer(cols * rows); // reserve space for 
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			dict.at<uint8_t>(i, j) = colmap::ReadBinaryLittleEndian<uint8_t>(&file);
		}
	}
	matcher::colmapVisualIndex<> visualIndex;
	visualIndex.Read(vocabPath);

	//read database
	colmap::Database geerardData(databasePath);
	auto images = geerardData.ReadAllImages();
	auto ids = geerardData.ReadVisualIDs(images[0].ImageId());
	std::cout << "\ndictionary path: " << ids.dictPath << "\ndictionary size: " << ids.dictsize;
	std::vector<std::pair<int, int>> imageid1, imageid2;
	for (int i = 0; i < 2; i++) {
		std::cout << "\nimageId: " << images[i].ImageId();
		auto currentIds = geerardData.ReadVisualIDs(images[i].ImageId()).ids;
		std::cout << "\ncolmap descriptor visual ids: \n";
		for (int j = 0; j < currentIds.rows(); j++) {
			if (i == 0) {
				imageid1.push_back({ currentIds(j,0),currentIds(j,1) });
			}
			if (i == 1) {
				imageid2.push_back({ currentIds(j,0),currentIds(j,1) });
			}
		}
		std::cout << "\ncolmap matches size: " << currentIds.rows();
		//cv descriptor quantizing
		//retrieve descriptor
		
		auto cv_descriptor = geerardData.ReadDescriptors(images[i].ImageId());
		auto cv_keypoints = geerardData.ReadKeypoints(images[i].ImageId());
		/*cv::Mat cv_descrip(cv_descriptor.rows(),cv_descriptor.cols(),CV_32S);	
		eigen2cv(cv_descriptor, cv_descrip);
		auto matches = matcher::colmapFlannMatcher(cv_descrip, dict, 1);*/

		auto matches = visualIndex.FindWordIds(cv_descriptor, 1, 256, -1);


		std::cout << "\ncv descriptor visual ids: \n";
		for (int j = 0; j < 10; j++) {
			std::cout << j<<"  "<<matches(j,0) << " ";
			
		}
		std::cout << "\ncv matches size: " << matches.size();
		std::cout << "\n cv_descriptors size: " << cv_descriptor.rows();
		std::cout << "\n cv_keypoints size: " << cv_keypoints.size();
	}
	
}
inline void autoTest() {
	std::unordered_map<int, std::vector<size_t>> testmap;
	testmap[0].push_back(1);
	testmap[1].push_back(2);

	testmap[1] = std::vector<rsize_t>();

	for (auto val : testmap) {
		auto vet1 = val.second;
		for (auto vers : vet1) {
			std::cout << vers;
		}
	}
}
inline void recurKernelTestWithImage(int argc, const char* argv[]) {
	if (argc != 3) {
		std::cout << "ER:argc != 3";
		return;
	}
	
	std::string imgpath1 = argv[1];
	std::string imgpath2 = argv[2];

	//read kcenter and build kdtree
	cv::FileStorage reader;
	reader.open("D:\\thesis\\Visual-Coarse-Relocalization-for-AR-Navigation-App\\vsImgProject\\x64\\Release\\Result\\UKB_covdet_1000_kmeansCenter.yml", cv::FileStorage::READ);
	if (!reader.isOpened()) { std::cout << "failed to open the kcenter file" << std::endl; }
	//read kcenters
	cv::Mat kCenters;
	reader["kcenters"] >> kCenters;
	reader.release();
	matcher::kdTree kdtreematcher(kCenters);

	//read image and build graph on it
	auto img1 = cv::imread(imgpath1, cv::IMREAD_GRAYSCALE);
	auto img2 = cv::imread(imgpath2, cv::IMREAD_GRAYSCALE);
	std::vector<cv::KeyPoint> kpts1, kpts2;
	cv::Mat descrip1, descrip2;
	extractor::covdetSIFT(img1, descrip1, kpts1);
	extractor::covdetSIFT(img2, descrip2, kpts2);

	//do matching
	auto matches1 = kdtreematcher.search(descrip1);
	auto matches2 = kdtreematcher.search(descrip2);
	std::cout << "matches1: " << matches1.size() << std::endl;
	std::cout << "matches2: " << matches2.size() << std::endl;
	//build graph
	igraph_t graph1, graph2;
	graph::build(matches1, kpts1, graph1);
	graph::build(matches2, kpts2, graph2);

	//kernel object
	kernel::recurRobustKel kernelObj(3, params::centers);
	
	std::unordered_map<int, std::vector<float>> scores;
	std::unordered_map<int, std::vector<size_t>> inv1, inv2;
	kernelObj.robustKernelCom(graph1,graph2,scores, inv1, inv2);

	//compute the total score
	float totalScores = 0;
	for (auto val : scores) {
		std::cout << val.first << ": ";
		for (auto j : val.second) {
			std::cout << j << " ";
		}
		totalScores += val.second.back();
		std::cout << "\n";
	}
	std::cout << "total score: " << totalScores << std::endl;

}
inline void recurKernelTest() {
	//define testing graphs
	auto sTime = clock();
	igraph_t testGraph;
	igraph_i_set_attribute_table(&igraph_cattribute_table);
	igraph_real_t edges[] = { 0,1,1,2,1,6,2,3,3,4,3,5,6,7,6,8,8,9,8,10 };
	igraph_real_t label1[] = { 0,2,1,3,4,5,3,6,7,8,9 };
	igraph_vector_t e1, lab1;

	igraph_vector_view(&e1, edges, sizeof(edges) / sizeof(double));
	igraph_vector_view(&lab1, label1, sizeof(label1) / sizeof(double));
	igraph_create(&testGraph, &e1, 0, IGRAPH_UNDIRECTED);
	SETGAS(&testGraph, "name", "testGraph1");
	SETGAN(&testGraph, "n_vertices", sizeof(label1) / sizeof(double));
	SETVANV(&testGraph, "label", &lab1);

	igraph_t testGraph2;
	igraph_vector_t e2, lab2;
	igraph_real_t edges2[] = { 0,2,1,2,2,3,3,4,4,5,5,6,5,7 };
	igraph_real_t label2[] = { 1,3,0,2,3,6,8,7 };

	igraph_vector_view(&e2, edges2, sizeof(edges2) / sizeof(double));
	igraph_vector_view(&lab2, label2, sizeof(label2) / sizeof(double));
	igraph_create(&testGraph2, &e2, 0, IGRAPH_UNDIRECTED);
	SETGAS(&testGraph2, "name", "testGraph2");
	SETGAN(&testGraph2, "n_vertices", sizeof(label2) / sizeof(double));
	SETVANV(&testGraph2, "label", &lab2);

	//compute the recursive kernel value
	kernel::recurRobustKel kernelObj(3, 10);
	kernelObj.push_back(testGraph);
	kernelObj.push_back(testGraph2);
	std::unordered_map<int, std::vector<float>> scores;
	std::unordered_map<int, std::vector<size_t>> inv1, inv2;
	kernelObj.robustKernelCom(0,1,scores,inv1,inv2);
	
	for (auto val : scores) {
		std::cout << val.first << ": ";
		for (auto j : val.second) {
			std::cout << j << " ";
		}
		std::cout << "\n";
	}
	
}
inline void covdetTest() {
	std::string img1 = "D:\\thesis\\datasets\\ukbench\\full\\ukbench00016.jpg";
	std::string kcenterPath = "D:\\thesis\\Visual-Coarse-Relocalization-for-AR-Navigation-App\\vsImgProject\\x64\\Release\\Result\\UKB_covdet_1000_kmeansCenter.yml";
	cv::FileStorage reader;
	reader.open(kcenterPath, cv::FileStorage::READ);
	if (!reader.isOpened()) { std::cout << "failed to open the kcenter file" << std::endl; return; }

	//read kcenters
	cv::Mat kCenters;
	reader["kcenters"] >> kCenters;
	reader.release();
	
	
	cv::Mat imgGray = cv::imread(img1, cv::IMREAD_GRAYSCALE);
	if (!imgGray.data) {
		std::cout << "img load failed!";
		return;
	}
	cv::Mat imgColor = cv::imread(img1), imgResize;
	cv::resize(imgColor, imgResize, cv::Size(), params::imgScale, params::imgScale, cv::INTER_AREA);
	std::vector<cv::KeyPoint> siftKpts, covdetKpts;
	cv::Mat siftDescrips,covdetDescrips;
	extractor::covdetSIFT(imgGray, covdetDescrips, covdetKpts);

	//use vlfeat native sift extractor
	extractor::vlimg_descips_compute_simple(imgGray, siftDescrips, siftKpts);
	std::cout << "siftKeypoints size: " << siftKpts.size()<<std::endl;
	std::cout << "covdetKeypoints size: " << covdetKpts.size()<<std::endl;
	//draw the images
	cv::Mat outcovdetImg, outsiftImg;
	/*cv::drawKeypoints(imgResize, covdetKpts, outcovdetImg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::drawKeypoints(imgResize, siftKpts, outsiftImg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	
	cv::namedWindow("covdet keypoints", cv::WINDOW_NORMAL);
	cv::imshow("covdet keypoints", outcovdetImg);
	std::cout << "covdet size: " << outcovdetImg.size()<<std::endl;
	cv::namedWindow("sift keypoints", cv::WINDOW_NORMAL);
	std::cout << "sift size: " << outsiftImg.size() << std::endl;
	cv::imshow("sift keypoints", outsiftImg);
	cv::waitKey();*/


	matcher::kdTree matchers(kCenters);
	auto bestMatches = matchers.search(covdetDescrips);
	auto bestMatchesSift=matchers.search(siftDescrips);
	//find the corresponding matches
	
	std::cout << "covdet best matches size: " << bestMatches.size() << std::endl;
	std::cout<< "sift best matches size: " << bestMatchesSift.size() << std::endl;
	//print descriptors
	/*for (int i = 0; i < 10; i++) {
		std::cout << "covdetDescrip " << i << ": \n" << covdetDescrips.row(bestMatches[i].queryIdx);
		std::cout << "siftDescrip " << i << ": \n" << siftDescrips.row(bestMatches[i].trainIdx);
		std::cout << "difference: \n" << covdetDescrips.row(bestMatches[i].queryIdx) - siftDescrips.row(bestMatches[i].trainIdx);
	}*/

	/*cv::drawMatches(imgResize, covdetKpts, imgResize, siftKpts, bestMatches, imgColor, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow("matches keypoints", cv::WINDOW_NORMAL);
	cv::imshow("matches keypoints", imgColor);
	cv::waitKey(); */
}

#endif
