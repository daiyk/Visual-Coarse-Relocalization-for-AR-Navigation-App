#include <iostream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
const char *keys =
    "{ help h |                  | Print help message. }"
    "{ input1 | box.png          | Path to input image 1. }"
    "{ input2 | box_in_scene.png | Path to input image 2. }";


void feature_lab(){

}
int main(int argc, char *argv[])
{
    CommandLineParser parser(argc, argv, keys);
    Mat img1 = imread(samples::findFile(parser.get<String>("input1")), IMREAD_GRAYSCALE);
    Mat img2 = imread(samples::findFile(parser.get<String>("input2")), IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty())
    {
        cout << "Could not open or find the image!\n"
             << endl;
        parser.printMessage();
        return -1;
    }
    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 2000;
    Ptr<SURF> detector = SURF::create(minHessian);
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
    //-- Step 2: Matching descriptor vectors with a brute force matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches,2);

    //define the threshold
    const float threshold_const = 0.7;
    std::vector<DMatch> in_matches;
    //define the good and bad matches
    for(int i=0;i<knn_matches.size();i++){
       
            if(knn_matches[i][0].distance<threshold_const*knn_matches[i][1].distance){
                in_matches.push_back(knn_matches[i][0]);
            }
    }
    //-- Draw matches
    Mat img_matches; // just stores the generated images
    drawMatches(img1, keypoints1, img2, keypoints2, in_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //-- Show detected matches
    cout<<"dWescirptor size"<<descriptors1.size<<endl<<"desciptor2 size"<<descriptors2.size<<endl;
    cout<<"matches's size"<<knn_matches.size();
    cout<<"matches:"<<in_matches.size()<<endl;
    namedWindow("Matches",WINDOW_NORMAL);
    // imshow("Matches", img_matches);
    waitKey();
    return 0;
}
#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif