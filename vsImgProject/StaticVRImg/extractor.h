#pragma once

#ifndef _EXTRACTOR_H
#define _EXTRACTOR_H
#include <vector>
#include <opencv2/core.hpp>
namespace extractor {
	// opencv pipeline to do extraction of keypoints and descriptors 
	void openCVimg_descips_compute(std::vector<std::string>& paths, cv::Mat& allDescripts, std::vector<cv::KeyPoint>& keypoints);

	// vlfeat pipleline to do extraction of keypoints and descriptors
	void vlimg_descips_compute(std::vector<std::string>& paths, cv::Mat& allDescripts, std::vector<cv::KeyPoint>& cv_keypoints);

	//simple single image descriptor computing
	//img1: must be grayImg
	void vlimg_descips_compute_simple(cv::Mat& img1, cv::Mat& Descripts, std::vector<cv::KeyPoint>& cv_keypoints);
}
#endif // ! _DESCRIP_EXT_H

