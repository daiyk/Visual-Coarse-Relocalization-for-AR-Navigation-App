#pragma once
#ifndef _FEATURETEST_H
#define _FEATURETEST_H
#include <StaticVRImg/fileManager.h>
#include <StaticVRImg/extractor.h>
#include <StaticVRImg/matcher.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using params = fileManager::parameters;
inline void covdetTest() {
	std::string img1 = "D:\\thesis\\datasets\\ukbench\\full\\ukbench00016.jpg";
	std::string kcenterPath = "D:\\thesis\\Visual-Coarse-Relocalization-for-AR-Navigation-App\\vsImgProject\\x64\\Release\\Result\\UKB_vlfeat_1000_kmeansCenter.yml";
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


	/*matcher::kdTree matchers(kCenters);*/
	//find the corresponding matches
	auto bestMatches = matcher::opencvFlannMatcher(kCenters,covdetDescrips);
	auto bestMatchesSift = matcher::opencvFlannMatcher(kCenters, siftDescrips);
	/*auto bestMatches = matcher::opencvFlannMatcher(siftDescrips, covdetDescrips);*/
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
