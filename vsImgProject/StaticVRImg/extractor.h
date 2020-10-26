#pragma once

#ifndef _EXTRACTOR_H
#define _EXTRACTOR_H
#include <colmap/util/bitmap.h>
#include <vector>
#include <opencv2/core.hpp>

namespace extractor {
	// opencv pipeline to do extraction of keypoints and descriptors 
	void openCVimg_descips_compute(std::vector<std::string>& paths, cv::Mat& allDescripts, std::vector<cv::KeyPoint>& keypoints);

	// vlfeat pipleline to do extraction of keypoints and descriptors
	void vlimg_descips_compute(std::vector<std::string>& paths, cv::Mat& allDescripts, std::vector<cv::KeyPoint>& cv_keypoints);

	//simple single image descriptor computing
	//img: input img must be grayscale
	//kpts: container for detected keypoint
	//descriptors: container for SIFT descriptors
	void vlimg_descips_compute_simple(cv::Mat& img1, cv::Mat& Descripts, std::vector<cv::KeyPoint>& cv_keypoints, colmap::Bitmap* bitmap = nullptr);

	/*void siftGPU_descips_compute_simple(std::vector<colmap::Bitmap> queryImgs, std::vector<colmap::FeatureKeypoints>& kpts, std::vector<colmap::FeatureDescriptors>& descripts);*/
	//covdet implementation of SIFT
	//img: input img must be grayscale
	//kpts: container for detected keypoint
	//descriptors: container for SIFT descriptors
	void covdetSIFT(cv::Mat& img, cv::Mat& descriptors, std::vector<cv::KeyPoint>& kpts);


	
}
#endif // ! _DESCRIP_EXT_H

