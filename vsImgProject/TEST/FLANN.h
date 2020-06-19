#include <iostream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "StaticVRImg/extractor.h"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

void FLANN(Mat& img1, Mat& img2, std::vector<DMatch> good_matches)
{
    //-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
    //SIFT for feature extraction
    Mat descript1, descript2;
    std::vector<KeyPoint> kpts1, kpts2;
    extractor::vlimg_descips_compute_simple(img1, descript1, kpts1);
    extractor::vlimg_descips_compute_simple(img2, descript2, kpts2);
    
    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch(descript1, descript2, knn_matches, 2);
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    ;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    Mat img_matches;
}
double FLANNScore(std::vector<DMatch>& good_matches) {
    //iterate over the good matches and compute the scores
    double distance = 0.0;
    for (auto it:good_matches) {
        distance += it.distance;
    }
    return distance / good_matches.size();
}

#else
int FLANN()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif