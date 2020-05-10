#pragma once
#ifndef _CONSTANDTYPES_H
#define _CONSTANDTYPES_H
#include <limits>
#include <opencv2/core.hpp>
namespace constandtypes{

	// below will be transformed to json file.
	const int octave = 8;       // number of octave used in the sift detection
	const int noctaveLayer = 3; // scale layers per octave
	const int octave_start = 1; // learning start from 1th octave, -1 for more details
	const double sigma_0 = 1.6; // sigma for the #0 octave
	const int centers = 200;    // k-means center detection, defines the number of centers
	const int numOfAttemp = 3; //times of try to compute the center for each cluster, five times to choose the best one
	const int numOfItera = 20;
	const double accuracy = 1e-3;

	//OpenCV relevent setting
	const auto criteria = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, numOfItera, accuracy); //stop criteria, COUNT means number of iter, EPS means convergence accuracy
	const float MATCH_THRES = 0.7; //define the threshold for matching 

	//matching setting
	const int numOfNN = 2;
}
#endif


////graph building constants
	//int minNumDeg = 0;
	//int maxNumDeg = -1;
	//double radDegLim = std::numeric_limits<double>::infinity();
