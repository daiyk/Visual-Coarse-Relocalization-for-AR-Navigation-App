#pragma once
#ifndef _HELPER_H
#define _HELPER_H
#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS
#include <opencv2/core.hpp>
#include <ctime>

//helper function to get current date
std::string dateTime() {
    std::time_t t = std::time(0);   // get time now
    std::tm* now = std::localtime(&t);
    return std::to_string(now->tm_hour) + std::to_string(now->tm_min) + std::to_string(now->tm_sec);
}

//unpack octave number, copy from opencv source code https://github.com/opencv/opencv_contrib/blob/bebfd717485c79644a49ac406b0d5f717b881aeb/modules/xfeatures2d/src/sift.cpp#L214-L220
void unpackOctave(const cv::KeyPoint& kpt, int& octave, int& layer, float& scale)
{
    octave = kpt.octave & 255;
    layer = (kpt.octave >> 8) & 255;
    octave = octave < 128 ? octave : (-128 | octave);
    scale = octave >= 0 ? 1.f / (1 << octave) : (float)(1 << -octave);
}
#endif
