#include "FLANN.h"
#include "extractor.h"
#include "fileManager.h"
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
using namespace cv;
using std::cout;
using std::endl;
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"

void FLANN::FLANNMatch(cv::Mat& descripts1, cv::Mat& descripts2, std::vector<cv::DMatch>& good_matches) {
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch(descripts1, descripts2, knn_matches, 2);
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < fileManager::parameters::MATCH_THRES * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
}
void FLANN::FLANNImgsMatch(Mat& grayImg1, Mat& grayImg2, std::vector<DMatch>& good_matches)
{
    //-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
    //SIFT for feature extraction
    Mat descript1, descript2;
    std::vector<KeyPoint> kpts1, kpts2;
    extractor::vlimg_descips_compute_simple(grayImg1, descript1, kpts1);
    extractor::vlimg_descips_compute_simple(grayImg2, descript2, kpts2);

    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    FLANNMatch(descript1, descript2, good_matches);
}
#else
void FLANN::FLANNMatch(Mat& img1, Mat& img2, std::vector<DMatch> good_matches)
{
    std::cout << "FLANNMatch: Terminated: this program needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif
double FLANN::FLANNScore(std::vector<DMatch>& good_matches) {
    //iterate over the good matches and compute the scores
    double distance = 0.0;
    for (auto it : good_matches) {
        distance += it.distance;
    }
    return distance / good_matches.size();
}