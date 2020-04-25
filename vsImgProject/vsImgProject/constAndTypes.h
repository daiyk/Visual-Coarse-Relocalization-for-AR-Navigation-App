#pragma once

#ifndef _CONSTANDTYPES_H
#define _CONSTANDTYPES_H
#include <opencv2/core.hpp>
namespace constandtypes {

	//args enum for argcs read
	enum class ArgType {
		TOOL_OPENCV,
		TOOL_VLFEAT,
		TOOL_OPENCV_AND_VLFEAT,
		MODE_TRAIN,
		MODE_MATCHING,
		MODE_DEMO,
		DEFAULT
	};

	const int octave = 8;       // number of octave used in the sift detection
	const int noctaveLayer = 3; // scale layers per octave
	const int octave_start = 1; // learning start from 1th octave, -1 for more details
	const double sigma_0 = 1.6; // sigma for the #0 octave
	const int centers = 200;    // k-means center detection, defines the number of centers
	const int numOfAttemp = 3; //times of try to compute the center for each cluster, five times to choose the best one
	const int numOfItera = 20;
	const double accuracy = 1e-3;

	//OpenCV relevent setting
	const auto criteria = TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, numOfItera, accuracy); //stop criteria, COUNT means number of iter, EPS means convergence accuracy
	const float MATCH_THRES = 0.7; //define the threshold for matching 

	//matching setting
	const int numOfNN = 2;
}


#endif
