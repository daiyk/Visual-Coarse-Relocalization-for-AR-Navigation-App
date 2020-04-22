#include <iostream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/persistence.hpp>
#include "Shlwapi.h"
#include "omp.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
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
const int dict_size = 200; //kmeans dictionary size
const auto criteria = TermCriteria(TermCriteria::COUNT|TermCriteria::EPS, 20, 1e-3); //stop criteria, COUNT means number of iter, EPS means convergence accuracy
const int attempts = 3; //times of try to compute the center for each cluster, five times to choose the best one
float MATCH_THRES = 0.6;

//unpack octave number, copy from opencv source code https://github.com/opencv/opencv_contrib/blob/bebfd717485c79644a49ac406b0d5f717b881aeb/modules/xfeatures2d/src/sift.cpp#L214-L220
void unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale)
{
    octave = kpt.octave & 255;
    layer = (kpt.octave >> 8) & 255;
    octave = octave < 128 ? octave : (-128 | octave);
    scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
}

//generate two files: (optionally) recording keypoints and kcenters
void write_to_file(std::string name, std::vector<KeyPoint> &kpts, Mat &kCenters){
    std::ofstream CSVOutput;
    int nKpts = kpts.size();
    CSVOutput.open(std::string(name+"_"+std::to_string(nKpts)+".csv"),std::fstream::out|std::fstream::app);

    //keypoint write to file
    if(!kpts.empty()){
        //input stream for headers
        CSVOutput<<"Oritentation"<<","<<"Octave"<<","<<"layer"<<","<<"pt.x"<<","<<"pt.y"<<","<<"scale"<<"\n";

        //write data to the file
        int numOkpt = kpts.size();
        int max_noctave=0;
        for(int i=0;i<numOkpt;i++){
            auto kp = kpts[i];
            int noctave, nlayer;
            float vscale;
            if(noctave>max_noctave){
                max_noctave = noctave;
            }
            unpackOctave(kp,noctave,nlayer,vscale);
            //unpack keypoints value
            CSVOutput<<kp.angle<<","
                    <<noctave<<","
                    <<nlayer<<","
                    <<kp.pt.x<<","
                    <<kp.pt.y<<","
                    <<vscale<<"\n";
        }
        CSVOutput.close();
        cout<<"keypoints store finish with octave num: "<<max_noctave<<endl;
    }

    //write kcenters to files
    if(!kCenters.empty()){
        cv::FileStorage filewriter(name+"_kmeansCenter.yml",cv::FileStorage::WRITE);
        filewriter<<"kcenters"<<kCenters;
        cout<<"visual words are written to file kmeansCenter.yml......"<<endl;
    }
    
}

//SIFT detection training imgs and kmeans learning for clustering
void img_descips_compute(std::vector<String> &paths, std::vector<KeyPoint> &keypoints, Mat &kCenters)
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
        for(size_t i=0;i<num_imgs;i=i+1){
            // int minHessian = 400; // for SURF detector only
            Mat grayImg;
            Ptr<SIFT> detector = SIFT::create();
            if(!fs::exists(fs::path(paths[i]))){
                cout<<"Warning: "<<paths[i]<<"does not exist, pass the image"<<endl;
                continue;
            }
            cvtColor(imread(paths[i]),grayImg,COLOR_BGR2GRAY);

            //surf to detect and compute
            Mat descriptors;
            std::vector<KeyPoint> kpts;
            detector->detectAndCompute(grayImg, noArray(), kpts, descriptors);
            
            //add to the total statistics, descriptors and keypoints
            #pragma omp critical
            {
                allDescrips.push_back(descriptors);
                keypoints.reserve(keypoints.size()+kpts.size());
                keypoints.insert(keypoints.end(),kpts.begin(),kpts.end());

            }
            
        }
    }   
    clock_t sTime=clock();
    cout<<"start k-means learning..."<<endl;
    Mat outImg,grayImg;
    cvtColor(imread(paths[0]),grayImg,COLOR_BGR2GRAY);
    outImg = grayImg;
    drawKeypoints(grayImg,keypoints,outImg,Scalar::all(-1),DrawMatchesFlags::DRAW_OVER_OUTIMG);
    imshow("test Img",outImg);
    waitKey();
    Mat labels; //stores the trained labels
    //k-means 
    kmeans(allDescrips,dict_size,labels,criteria,attempts,KMEANS_PP_CENTERS,kCenters);
    cout<<"successfully produce cluster centers MAT with size: "<<kCenters.rows<<endl;
    cout<<"-> kmeans learning spent "<<(clock()-sTime)/double(CLOCKS_PER_SEC)<<" sec"<<endl;
    
}

//detect test image keypoints and descriptors and write out statistics
void test_image_lab(Mat &test_img, Mat &kCenters, Mat &outImg, std::vector<KeyPoint> &keypoints){
    //extract the features from test image
    Ptr<SIFT> siftdetect = SIFT::create();

    //create statistic for each features

    Mat descriptors,grayImg;
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

    //store the matching statistics
    test_img.copyTo(outImg); //fill outImg with original img
    std::vector<std::vector<KeyPoint>> kpts_feat;
    cv::RNG rng(123);
    for(auto it=hist_centers.begin();it!=hist_centers.end();it++){
        Scalar color = Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
        std::vector<KeyPoint> kp_loc;

        //accumulate same class kp
        for(auto it2=it->begin();it2!=it->end();it2++){
            kp_loc.push_back(keypoints[*it2]);
        }
        kpts_feat.push_back(kp_loc);    //stores statistics for each keywords
        //draw repeatly on the existing images
        drawKeypoints(test_img,kp_loc,outImg,color,DrawMatchesFlags::DRAW_OVER_OUTIMG);

    }

    std::ofstream CSVOutput;
    int nKpts = keypoints.size();
    CSVOutput.open(std::string(std::to_string(nKpts)+".csv"),std::fstream::out|std::fstream::app);

    //input stream for headers
    CSVOutput<<"feature No."<<","<<"Num."<<"\n";

    //write data to the file
    for(int i=0;i<kpts_feat.size();i++){
        CSVOutput<<i<<","<<kpts_feat[i].size()<<"\n";
    }
    CSVOutput.close();
    cout<<"statistics of keypoints to features finished....."<<endl;
}

//read paths for train and test imgs
bool readFiles(int argc, char* argv[],std::vector<std::string>& trainFilePaths, std::vector<std::string> &testFilePaths) {
    std::fstream fileRead;
    //open files with link of images
    CommandLineParser parser(argc, argv, keys);
    fs::path imgs_path = parser.get<String>("path");
    Mat img1 = imread(samples::findFile(parser.get<String>("input1"), false), IMREAD_GRAYSCALE);
    Mat img2 = imread(samples::findFile(parser.get<String>("input2"), false), IMREAD_GRAYSCALE);
    /**
        TODO code for img1 and img2 comparison
    **/

    /**
     * if path is provided, we compute the correspondence
     *
     * */
    if (!imgs_path.empty() && img1.empty() && img2.empty()) {
        if (!imgs_path.empty() && !fs::exists(imgs_path)) {
            cout << "ERROR: provided imgs path doesn't exist!" << endl;
            return false;
        }

        //find the train imgs
        fs::path train_path = imgs_path;
        fs::path test_path = imgs_path;
        train_path /= "train";
        test_path /= "test";
        cout << "Current training images path: " << train_path << endl;
        cout << "Current testing images path: " << test_path << endl;
        if (!fs::exists(train_path)) {
            std::cerr << "ERROR: train subfolder of provided path doesn't exist!" << endl;
            return false;
        }
        else if (!fs::exists(train_path)) {
            std::cerr << "WARNING: test subfolder for provided path doesn't exist! training still in process......" << endl;
        }
        else {
            for (const auto& entry : fs::directory_iterator(test_path)) {
                std::string::size_type idx;
                idx = entry.path().string().rfind('.');
                if (idx != std::string::npos)
                {
                    std::string extension = entry.path().string().substr(idx + 1);
                    if (extension == "jpg") {
                        testFilePaths.push_back(entry.path().string());
                        cout << "test img is added and found at: " << entry.path().string() <<"......"<< endl;
                    }
                }
            }
        }

        for (const auto& entry : fs::directory_iterator(train_path)) {
            std::string::size_type idx;
            idx = entry.path().string().rfind('.');
            if (idx != std::string::npos)
            {
                std::string extension = entry.path().string().substr(idx + 1);
                if (extension == "jpg") {
                    trainFilePaths.push_back(entry.path().string());
                    cout << "img is added and found at: " << entry.path().string() << "......"<<endl;
                }
            }
        }
        return true;
    }
    /**
     * if both are provided, raise error
     * 
     * */
    
    if (img1.empty() || img2.empty())
    {
        cout << "Could not open or find the image!\n"
             << endl;
        parser.printMessage();
        return -1;
    }
    if(imgs_path.empty() && !img1.empty() && !img2.empty()){
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

        std::vector<DMatch> in_matches;
        //define the good and bad matches
        for(int i=0;i<knn_matches.size();i++){
        
                if(knn_matches[i][0].distance<MATCH_THRES*knn_matches[i][1].distance){
                    in_matches.push_back(knn_matches[i][0]);
                }
        }
        //-- Draw matches
        Mat img_matches; // just stores the generated images
        drawMatches(img1, keypoints1, img2, keypoints2, in_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        //-- Show detected matches
        cout<<"descirptor size :"<<descriptors1.size<<endl<<"desciptor2 size"<<descriptors2.size<<endl;
        cout<<"matches's size :"<<knn_matches.size();
        cout<<"matches size: "<<in_matches.size()<<endl;
        namedWindow("Matches",WINDOW_NORMAL);
        imshow("Matches", img_matches);
        imwrite("test_matching.jpg",img_matches);
        waitKey();
        if(!img1.empty() && !img2.empty() && !imgs_path.empty()){
            cout<<"COMFLICT ERROR: Please EITHER provide path or provide input 1/2."<<endl;
            return false;
        }
        return true;
    }
    return false;
    //TODO: function for read other types of arg inputs
}

//main function for train kmeans clustering
bool train(std::vector<String> trainPaths, Mat &kCenters)
{

        std::vector<KeyPoint> kpts;
        try{
            img_descips_compute(trainPaths,kpts,kCenters);
            // vlimg_descips_compute(paths,kCenters);
        }
        catch(std::invalid_argument &e){
            cout<<"Exception: "<<e.what()<<endl;
            return false;
        }

        //write out kecenters
        std::vector<KeyPoint> shadowkpts;
        write_to_file("train_statisic",shadowkpts,kCenters);
    
    return true;
}

//main function for detecting test img keypoints
bool test(std::vector<String> testPaths, Mat &kCenters){

    if(testPaths.empty()){
        cout<<"Test img path empty skip test img......" << endl;
        return false;
    }
    Mat out_img, kpts_img, test_img = imread(testPaths[0]);
    std::vector<KeyPoint> kpts;

    test_image_lab(test_img,kCenters,out_img,kpts);
    std::vector<std::vector<KeyPoint>> octave_kpts;
    octave_kpts.resize(8);
    
    for(int i = 0;i<kpts.size();i++){
        int noctave, nlayer;
        float scale; 
        unpackOctave(kpts[i],noctave,nlayer,scale);
        octave_kpts[noctave+1].push_back(kpts[i]);
    }
    
    //try to show different octave's image

    imshow("Test result",out_img);
    imwrite("Test_result.jpg",out_img);
    drawKeypoints(test_img,kpts,kpts_img,Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    imshow("kpts Img", kpts_img);
    imwrite("kpts Img.jpg",kpts_img);
    for(int i = 0;i<octave_kpts.size();i++)
    {
        if(octave_kpts.size()!=0){
            Mat octave_img;
            cout<<"octave "+std::to_string(i-1)+" with size"<< octave_kpts[i].size()<<endl;
            drawKeypoints(test_img,octave_kpts[i],octave_img,Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            imwrite("test_octave_"+std::to_string(i-1)+".jpg",octave_img);
        }
    }
    Mat shadowkcenter;
    write_to_file("test_image",kpts,shadowkcenter);
    waitKey();
    return true;
}

int main(int argc, char *argv[]){
    
    //readFiles from arg paths
    Mat kCenters;
    std::vector<String> trainPaths,testPaths;
    readFiles(argc,argv,trainPaths,testPaths);
    //start train
    train(trainPaths,kCenters);
    
    FileStorage ReadKcenters("kmeansCenter.yml",cv::FileStorage::READ);
    
    
    // ReadKcenters["kcenters"]>>kCenters;
    // std::cout<<"read kcenters success......"<<endl;
    // //test image
    // test(testPaths,kCenters);
    return 0;
}
#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif


// std::fstream fileRead;
//     //open files with link of images
//     fileRead.open("D:\\thesis\\Visual-Coarse-Relocalization-for-AR-Navigation-App\\imgs\\ground-truth-database-images-slice2.txt");

//     if(!fileRead.fail()){
//         std::string line;
//         std::vector<String> sentence;
//         while(std::getline(fileRead,line)){
//             std::istringstream iss(line);
//             std::copy(std::istream_iterator<String>(iss),
//             std::istream_iterator<String>(),
//             std::back_inserter(sentence)
//             );
//         }
//     }
//     else
//     {
//         cout<<"Provided files open failed"<<endl;
//     }
    
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