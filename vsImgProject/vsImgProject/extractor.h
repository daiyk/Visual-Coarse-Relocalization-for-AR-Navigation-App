#pragma once

#ifndef _EXTRACTOR_H
#define _EXTRACTOR_H
#include <vector>
#include <opencv2/core.hpp>
using namespace cv;

// opencv pipeline to do extraction of keypoints and descriptors 
void openCVimg_descips_compute(std::vector<std::string>& paths, Mat& allDescripts, std::vector<KeyPoint>& keypoints);

// vlfeat pipleline to do extraction of keypoints and descriptors
void vlimg_descips_compute(std::vector<std::string>& paths, Mat& allDescripts, std::vector<KeyPoint>& cv_keypoints);

#endif // ! _DESCRIP_EXT_H

