#pragma once
#ifndef _CLUSTER_H
#define _CLUSTER_H
#include <opencv2/core.hpp>

void openCV_visual_words_compute(cv::Mat& allDescripts, cv::Mat& kCenters);
void vl_visual_word_compute(cv::Mat& allDescrip, cv::Mat& kCenters);

#endif