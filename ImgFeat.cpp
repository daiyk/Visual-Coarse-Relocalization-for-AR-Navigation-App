#include <iostream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc.hpp"
#include "Shlwapi.h"
#include "omp.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
extern "C" {
    #include "vl/vlad.h"
    #include "vl/sift.h"
    #include "vl/generic.h"
}



using namespace cv;
using namespace cv::xfeatures2d;
namespace fs = std::filesystem;
using std::cout;
using std::endl;
const char *keys =
    "{ help h |                  | Print help message. }"
    "{ path   |                  | Path to the image folder, not comp. with input1/2 }"
    "{ input1 |                  | Path to input image 1, not comp. with path }"
    "{ input2 |                  | Path to input image 2, not comp. with path }";

// const float MATCH_THRES = 0.7; //define the threshold for 

void vlimg_descips_compute(std::vector<String> &paths, Mat &kCenters)
{
    size_t num_imgs = paths.size();
    Mat allDescrips;
    if(num_imgs==0){; 
        throw std::invalid_argument("imgs folder is empty!");
    }
    omp_set_num_threads(6);
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for(size_t i=0;i<num_imgs;i=i+5){
            // int minHessian = 400; // for SURF detector only
            Mat grayImg;
            Mat grayImgFl;
            Mat imgDescrips;
            if(!fs::exists(fs::path(paths[i]))){
                cout<<"Warning: "<<paths[i]<<"does not exist, pass the image"<<endl;
                continue;
            }
            cvtColor(imread(paths[i]),grayImg,COLOR_BGR2GRAY);

            //surf to detect and compute
            size_t width = grayImg.size().width;
            size_t height = grayImg.size().height;
            
            int octave =3;
            int noctaveLayer = 3;
            int octave_start = 1;
            double sigma_0 = 1.6; //sigma for the #0 octave

            auto vl_sift = vl_sift_new(width,height,octave,noctaveLayer,1);//define vl sift processor
            vl_sift_set_edge_thresh(vl_sift,10);
            vl_sift_set_peak_thresh(vl_sift,0.04);

            grayImg.convertTo(grayImgFl,CV_32F,1.0/255.0);
            
            float *img_ptr = grayImgFl.ptr<float>();
            
            if(!grayImgFl.isContinuous()){
                std::cerr<<"ERROR: when read "<<paths[i]<<" OpenCV finds uncontinuous address"<<endl;
                continue;
            }
            
            //go trough the loop of gaussian pyramid
            int result = vl_sift_process_first_octave(vl_sift,img_ptr);
            Mat1f allDescriptors;
            while(result!=VL_ERR_EOF){
                vl_sift_detect(vl_sift);

                //to get each keypoints
                int keyPtsNum = vl_sift_get_nkeypoints(vl_sift);

                const auto *keypoints = vl_sift_get_keypoints(vl_sift);

                //loop each keypoints and get orientation
                for(int i=0;i<keyPtsNum;i++){
                    double rot[4];
                    int nOrit = vl_sift_calc_keypoint_orientations(vl_sift,rot,&keypoints[i]);
                    //get the descriptors for each computed orientation
                    for(int j = 0;j<nOrit;j++){
                        float *curr_descriptor;
                        vl_sift_calc_keypoint_descriptor(vl_sift,curr_descriptor,&keypoints[i],rot[j]);
                        imgDescrips.push_back(Mat1f(1,128,curr_descriptor));
                    }
                    
                }
                result = vl_sift_process_next_octave(vl_sift);
            }
            
            //iterate through the 

            //define vlfeat 
            //for each octave to calculate the scale space
            //add to the total statistics, descriptors and keypoints
            #pragma omp critical
            {
                allDescrips.push_back(imgDescrips);
            }
         //delete sift
         vl_sift_delete(vl_sift);   
        }
    }   
    clock_t sTime=clock();
    cout<<"start k-means learning..."<<endl;

}


void img_descips_compute(std::vector<String> &paths, Mat &kCenters)
{
    size_t num_imgs = paths.size();
    Mat allDescrips;
    if(num_imgs==0){; 
        throw std::invalid_argument("imgs folder is empty!");
    }
    omp_set_num_threads(6);
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for(size_t i=0;i<num_imgs;i=i+5){
            // int minHessian = 400; // for SURF detector only
            Mat grayImg;
            Ptr<SIFT> detector = SIFT::create();
            if(!fs::exists(fs::path(paths[i]))){
                cout<<"Warning: "<<paths[i]<<"does not exist, pass the image"<<endl;
                continue;
            }
            cvtColor(imread(paths[i]),grayImg,COLOR_BGR2GRAY);

            //surf to detect and compute
            std::vector<KeyPoint> keypoints;
            Mat descriptors;
            detector->detectAndCompute(grayImg, noArray(), keypoints, descriptors);
            
            //add to the total statistics, descriptors and keypoints
            #pragma omp critical
            {
                allDescrips.push_back(descriptors);
            }
            
        }
    }   
    clock_t sTime=clock();
    cout<<"start k-means learning..."<<endl;
    //BoW trainer for k-means
    //nums of bags
    int dict_size = 200;
    Mat labels; //stores the trained labels
    auto criteria = TermCriteria(TermCriteria::COUNT|TermCriteria::EPS, 20, 1e-3); //stop criteria, COUNT means number of iter, EPS means convergence accuracy
    int attempts = 3; //times of try to compute the center for each cluster, five times to choose the best one

    //k-means 
    kmeans(allDescrips,dict_size,labels,criteria,attempts,KMEANS_PP_CENTERS,kCenters);
    cout<<"successfully produce cluster centers MAT with size: "<<kCenters.rows<<endl;
    cout<<"-> kmeans learning spent "<<(clock()-sTime)/double(CLOCKS_PER_SEC)<<" sec"<<endl;
    
}
void test_image_lab(Mat &test_img, Mat &kCenters, Mat &outImg){
    //extract the features from test image
    Ptr<SIFT> siftdetect = SIFT::create();

    Mat descriptors,grayImg;
    std::vector<KeyPoint> keypoints;
    cvtColor(test_img,grayImg,COLOR_BGR2GRAY);
    siftdetect->detectAndCompute(grayImg,noArray(),keypoints,descriptors);

    //match against the kmeans center descriptors
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    std::vector<DMatch> matchPairs;
    matcher->match(descriptors,kCenters,matchPairs);

    //create histogram vector
    std::vector<std::vector<int>> hist_centers;
    hist_centers.resize(kCenters.rows);

    for(auto it=matchPairs.begin();it!=matchPairs.end();it++){
        //find the corresponding container and store the id
        hist_centers[it->trainIdx].push_back(it->queryIdx);
    }

    //draw keypoints with color
    test_img.copyTo(outImg); //fill outImg with original img
    cv::RNG rng(123);
    for(auto it=hist_centers.begin();it!=hist_centers.end();it++){
        Scalar color = Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
        std::vector<KeyPoint> kp_loc;

        //accumulate same class kp
        for(auto it2=it->begin();it2!=it->end();it2++){
            kp_loc.push_back(keypoints[*it2]);
        }

        //draw repeatly on the existing images
        drawKeypoints(test_img,kp_loc,outImg,color,DrawMatchesFlags::DRAW_OVER_OUTIMG);
    }        
}

void feature_lab(std::vector<DMatch> matches_in){
// 

}
int main(int argc, char *argv[])
{
    std::fstream fileRead;
    //open files with link of images
    fileRead.open("D:\\thesis\\Visual-Coarse-Relocalization-for-AR-Navigation-App\\imgs\\ground-truth-database-images-slice2.txt");

    if(!fileRead.fail()){
        std::string line;
        std::vector<String> sentence;
        while(std::getline(fileRead,line)){
            std::istringstream iss(line);
            std::copy(std::istream_iterator<String>(iss),
            std::istream_iterator<String>(),
            std::back_inserter(sentence)
            );
        }
    }
    else
    {
        cout<<"Provided files open failed"<<endl;
    }
    
    CommandLineParser parser(argc, argv, keys);
    fs::path imgs_path = parser.get<String>("path");
    Mat img1 = imread(samples::findFile(parser.get<String>("input1"),false), IMREAD_GRAYSCALE);
    Mat img2 = imread(samples::findFile(parser.get<String>("input2"),false), IMREAD_GRAYSCALE);

    /**
     * if path is provided, we compute the correspondence
     * 
     * */
    if (!imgs_path.empty()&&img1.empty()&&img2.empty()){
        if(!imgs_path.empty()&&!fs::exists(imgs_path)){
            cout<<"ERROR: provided imgs path doesn't exist!"<<endl;
            return 0;
        }

        //find the train imgs
        fs::path train_path = imgs_path;
        train_path/="train";
        cout<<"Current training images path: "<<train_path<<endl;
        std::vector<String> paths;
        for(const auto &entry : fs::directory_iterator(train_path)){
            std::string::size_type idx;
            idx = entry.path().string().rfind('.');
            if(idx != std::string::npos)
            {
                std::string extension = entry.path().string().substr(idx+1);
                if(extension=="jpg"){
                    paths.push_back(entry.path().string());
                    cout<<"img is added and found at: "<<entry<<endl;
                }
            }       
        }
        Mat kCenters;
        try{
            img_descips_compute(paths, kCenters);
            // vlimg_descips_compute(paths,kCenters);
        }
        catch(std::invalid_argument &e){
            cout<<"Exception: "<<e.what()<<endl;
        }
        
        //find the test images
        fs::path test_path = imgs_path;
        test_path/="test";
        Mat test_img, out_img;
        std::vector<String> test_paths;
        for(const auto &entry : fs::directory_iterator(test_path)){
            std::string::size_type idx;
            idx = entry.path().string().rfind('.');
            if(idx != std::string::npos)
            {
                std::string extension = entry.path().string().substr(idx+1);
                if(extension=="jpg"){
                    test_paths.push_back(entry.path().string());
                    cout<<"test img is added and found at: "<<entry<<endl;
                }

            }       
        }
        test_img = imread(test_paths[0]);
        imshow("Test_img",test_img);
        test_image_lab(test_img,kCenters,out_img);
        imshow("Test_result",out_img);
        imwrite("Test_result.jpg",out_img);
        waitKey();
    }


    /**
     * if both are provided, raise error
     * 
     * */
    if(!img1.empty() && !img2.empty() && !imgs_path.empty()){
        cout<<"COMFLICT ERROR: Please EITHER provide path or provide input 1/2."<<endl;
    }

    
    VL_PRINT ("Hello world!") ;
    VL_PRINT("maybe successful");



    // if (img1.empty() || img2.empty())
    // {
    //     cout << "Could not open or find the image!\n"
    //          << endl;
    //     parser.printMessage();
    //     return -1;
    // }
    // //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    // int minHessian = 2000;
    // Ptr<SURF> detector = SURF::create(minHessian);
    // std::vector<KeyPoint> keypoints1, keypoints2;
    // Mat descriptors1, descriptors2;
    // detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    // detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
    // //-- Step 2: Matching descriptor vectors with a brute force matcher
    // // Since SURF is a floating-point descriptor NORM_L2 is used
    // Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    // std::vector<std::vector<DMatch>> knn_matches;
    // matcher->knnMatch(descriptors1, descriptors2, knn_matches,2);

    // std::vector<DMatch> in_matches;
    // //define the good and bad matches
    // for(int i=0;i<knn_matches.size();i++){
       
    //         if(knn_matches[i][0].distance<MATCH_THRES*knn_matches[i][1].distance){
    //             in_matches.push_back(knn_matches[i][0]);
    //         }
    // }
    // //-- Draw matches
    // Mat img_matches; // just stores the generated images
    // drawMatches(img1, keypoints1, img2, keypoints2, in_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    // //-- Show detected matches
    // cout<<"dWescirptor size"<<descriptors1.size<<endl<<"desciptor2 size"<<descriptors2.size<<endl;
    // cout<<"matches's size"<<knn_matches.size();
    // cout<<"matches:"<<in_matches.size()<<endl;
    // namedWindow("Matches",WINDOW_NORMAL);
    // // imshow("Matches", img_matches);
    // waitKey();
    return 0;
}
#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif