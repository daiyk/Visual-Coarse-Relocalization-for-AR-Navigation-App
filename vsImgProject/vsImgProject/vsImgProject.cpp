// vsImgProject.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

extern "C" {
    #include "vl/sift.h"
    #include "vl/generic.h"
    #include "vl/kmeans.h"
    #include "vl/kdtree.h"
}
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include "constAndTypes.h"
#include "fileop.h"
#include "matcher.h"
#include "extractor.h"
#include "cluster.h"
using namespace std;
using namespace cv;
namespace cAt = constandtypes;
int main(int argc, const char* argv[]) {
    VL_PRINT("!------- Feature detection program starts! ------!\n");
    VL_PRINT("!-------Your argument command line ------!\n");
    for (int i = 1; i < argc; i++) {
        cout << " ->" << argv[i] << endl;
    }
    std::vector<std::string> trainPaths, testPaths;
    fileop::ArgList readResult;
    Mat allDescrips, kCenters;
    std::vector<KeyPoint> keypoints;
    try
    {
        readResult = fileop::funTestRead(argc, argv, trainPaths, testPaths);
    }
    catch (const std::invalid_argument& msg)
    {
        cout << "Exception:" << msg.what() << endl;
        return 0;
    }

   
    cout << "->total files found: " << trainPaths.size() + testPaths.size() << endl;
    if (readResult.mode==cAt::ArgType::MODE_MATCHING) {
        VL_PRINT("!------- Start image matching with vlfeat ------!\n");
        matcher::kdTree(trainPaths[0],trainPaths[1]);
        VL_PRINT("!------- This program stop here ------!\n");
        return 0;
    }

    VL_PRINT("!------- Start feature detection with OpenCV and VLFeat ------!\n");
    if (readResult.tool==cAt::ArgType::TOOL_VLFEAT || readResult.tool==cAt::ArgType::TOOL_OPENCV_AND_VLFEAT) {
        VL_PRINT("!------- VLFeat ------!\n");
        clock_t sTime = clock();
        //start vlfeat sift feature detection
        try {
            extractor::vlimg_descips_compute(trainPaths, allDescrips, keypoints);
        }
        catch (std::invalid_argument& e) {
            cout << e.what() << endl;
            return 0;
        };


        //check if only test img feature detection
        if (readResult.mode==cAt::ArgType::MODE_TRAIN) {
            //train k-means classifier
            //free memory since keypoints during training is not useful 
            std::vector<KeyPoint>().swap(keypoints);
            cluster::vl_visual_word_compute(allDescrips, kCenters);
            std::vector<KeyPoint> kpts;
            fileop::write_to_file("vlfeat", kpts, kCenters);
        }
        //demo
        else if (readResult.mode==cAt::ArgType::MODE_DEMO) { 
            Mat outImg, purImg;
            outImg = imread(trainPaths[0]);
            purImg = outImg;
            cv::drawKeypoints(purImg, keypoints, outImg, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            namedWindow("vlfeat detection Img", WINDOW_NORMAL);
            imshow("vlfeat detection Img", outImg);
        }
        cout << "-> vlfeat SIFT detection / Kmeans learning totally spent " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " sec......" << endl;
    }

    //openCV pipeline
    if (readResult.tool==cAt::ArgType::TOOL_OPENCV_AND_VLFEAT || readResult.tool ==cAt::ArgType::TOOL_OPENCV) {
        VL_PRINT("!------- OpenCV ------!\n");
        clock_t sTime = clock();
        try {
            extractor::openCVimg_descips_compute(trainPaths, allDescrips, keypoints);
        }
        catch (std::invalid_argument& e) {
            std::cout << e.what() << endl;
            return 0;
        }

        //kmeans visual word computing by openCV
        if (readResult.mode==cAt::ArgType::MODE_TRAIN) {
            std::vector<KeyPoint>().swap(keypoints);
            cluster::openCV_visual_words_compute(allDescrips, kCenters);
            std::vector<KeyPoint> kpts;
            fileop::write_to_file("opencv", kpts, kCenters);
        }
        else if (readResult.mode==cAt::ArgType::MODE_DEMO) {
            Mat outImg, purImg;
            outImg = imread(trainPaths[0]);
            purImg = outImg;
            drawKeypoints(purImg, keypoints, outImg, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            namedWindow("opencv feat detect Img", WINDOW_NORMAL);
            imshow("opencv feat detect Img", outImg);
        }
        cout << "-> opencv SIFT detection / Kmeans learning totally spent " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " sec......" << endl;
        //write important data to file
    }
    if (readResult.mode == cAt::ArgType::DEFAULT || readResult.tool == cAt::ArgType::DEFAULT) {
        cout << "ERROR: unsupported arguments list" << endl;
    }
    VL_PRINT("!------- This program stop here ------!\n");
    waitKey();
    return 0;
}





//extern "C" {
//    //#include //"vl/vlad.h"
//    #include "vl/sift.h"
//    #include "vl/generic.h"
//    #include "vl/kmeans.h"
//    #include "vl/kdtree.h"
//}
//#include <iostream>
//#include <ctime>
//#include "opencv2/core.hpp"
//#ifdef HAVE_OPENCV_XFEATURES2D
//#include <opencv2/highgui.hpp>
//#include <opencv2/features2d.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/core/persistence.hpp>
//#include "Shlwapi.h"
//#include "omp.h"
//#include <filesystem>
//#include <fstream>
//#include <sstream>
//#include <algorithm>
//#include <iterator>
//#include <vector>
//#include <string>
//#include <iostream>
//#include <unordered_map>
//#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS
//using namespace cv;
//namespace fs = std::filesystem;
//using std::cout;
//using std::endl;
//using cmdType = std::unordered_map<String, bool>;
//const char *keys =
//     "{ help h |                  | Print help message. }"
//     "{ tool   |      both        | Lib used for SIFT, OpenCV or VLFeat, default both. }"
//     "{ path   |                  | Path to the training image folder, learning visual words, not compatable with input1/2 }"
//     "{ img    |                  | Path to single test input img, computing and visualize the keypoints and descriptor for the img }"
//     "{ input1 |                  | Image matching pairs, Path to input image 1, not comp. with path }"
//     "{ input2 |                  | Image matching pairs, Path to input image 2, not comp. with path }";
///*
//*   1. path: directory of the training images folders. Program detects and learns visual words(default 200 words) from a sets of image, save the words to "Result" folder;
//*   2. img: path to single test img. Program shows SIFT demo by detecting and computing features, visualize the result by OpenCV and vlfeat;
//*   3. input1/input2: path to images 1, 2 for matching demo. Program detects images features by SIFT and match them by FLANN, visualize the matching result.
//*
//* Note: 1, 2 and 3 are not compatable args; "JPG" is the only accepted format; set "-tool" option to "opencv"/"vlfeat" if only one lib is used; matching demo 3. only uses vlfeat. 
//    
// */
//const int octave = 8;       // number of octave used in the sift detection
//const int noctaveLayer = 3; // scale layers per octave
//const int octave_start = 1; // learning start from 1th octave, -1 for more details
//const double sigma_0 = 1.6; // sigma for the #0 octave
//const int centers = 200;    // k-means center detection, defines the number of centers
//const int numOfAttemp = 3; //times of try to compute the center for each cluster, five times to choose the best one
//const int numOfItera = 20;
//const double accuracy = 1e-3;
////k-means learning the cluters from the provided descriptors
//
////OpenCV relevent setting
//const auto criteria = TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, numOfItera, accuracy); //stop criteria, COUNT means number of iter, EPS means convergence accuracy
//const float MATCH_THRES = 0.7; //define the threshold for matching 
//
////matching setting
//const int numOfNN = 2;
//
//String dateTime() {
//    std::time_t t = std::time(0);   // get time now
//    std::tm* now = std::localtime(&t);
//    return std::to_string(now->tm_hour) + std::to_string(now->tm_min) + std::to_string(now->tm_sec);
//}
//void unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale)
//{
//    octave = kpt.octave & 255;
//    layer = (kpt.octave >> 8) & 255;
//    octave = octave < 128 ? octave : (-128 | octave);
//    scale = octave >= 0 ? 1.f / (1 << octave) : (float)(1 << -octave);
//}
//
//void write_to_file(std::string name, std::vector<KeyPoint>& kpts, Mat& kCenters) {
//    if (!fs::exists("Result")) {
//        fs::create_directories("Result");
//    }
//    std::ofstream CSVOutput;
//    int nKpts = kpts.size();
//    CSVOutput.open(std::string("Result/"+name + "_statistics_" + std::to_string(nKpts) +"_"+dateTime()+ ".csv"), std::fstream::out | std::fstream::app);
//
//    //keypoint write to file
//    if (!kpts.empty()) {
//        //input stream for headers
//        CSVOutput << "Oritentation" << "," << "Octave" << "," << "layer" << "," << "pt.x" << "," << "pt.y" << "," << "scale" << "\n";
//
//        //write data to the file
//        int numOkpt = kpts.size();
//        int max_noctave = 0;
//        for (int i = 0; i < numOkpt; i++) {
//            auto kp = kpts[i];
//            int noctave, nlayer;
//            float vscale;
//            unpackOctave(kp, noctave, nlayer, vscale);
//            if (noctave > max_noctave) {
//                max_noctave = noctave;
//            }
//            
//            //unpack keypoints value
//            CSVOutput << kp.angle << ","
//                << noctave << ","
//                << nlayer << ","
//                << kp.pt.x << ","
//                << kp.pt.y << ","
//                << vscale << "\n";
//        }
//        CSVOutput.close();
//        cout << "->keypoints store finish with octave num: " << max_noctave << endl;
//    }
//
//    //write kcenters to files
//    if (!kCenters.empty()) {
//        cv::FileStorage filewriter("Result/"+name + "_kmeansCenter.yml", cv::FileStorage::WRITE);
//        filewriter << "kcenters" << kCenters;
//        cout << "->visual words are written to file kmeansCenter.yml......" << endl;
//    }
//}
//void openCV_visual_words_compute(Mat &allDescripts,Mat &kCenters) {
//    clock_t sTime = clock();
//    cout << "start k-means learning..." << endl;
//    Mat labels; //stores the trained labels
//    //k-means 
//    kmeans(allDescripts, centers, labels, criteria, numOfAttemp, KMEANS_PP_CENTERS, kCenters);
//    cout << "->successfully produce cluster centers MAT with size: " << kCenters.rows << endl;
//    cout << "-> kmeans learning spent " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " sec......" << endl;
//}
//void openCVimg_descips_compute(std::vector<std::string>& paths, Mat& allDescripts, std::vector<KeyPoint>& keypoints)
//{
//    clock_t sTime = clock();
//    size_t num_imgs = paths.size();
//    if (num_imgs == 0) {
//        throw std::invalid_argument("imgs folder is empty!");
//    }
//    omp_set_num_threads(6);
//    #pragma omp parallel
//    {
//    #pragma omp for schedule(dynamic)
//        for (int i = 0; i < num_imgs; i = i + 1) {
//            // int minHessian = 400; // for SURF detector only
//            Mat grayImg;
//            Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();
//            if (!fs::exists(fs::path(paths[i]))) {
//                cout << "Warning: " << paths[i] << "does not exist, the image is ignored;" << endl;
//                continue;
//            }
//            cvtColor(imread(paths[i]), grayImg, COLOR_BGR2GRAY);
//
//            //surf to detect and compute
//            Mat descriptors;
//            std::vector<KeyPoint> kpts;
//            detector->detectAndCompute(grayImg, noArray(), kpts, descriptors);
//
//            //add to the total statistics, descriptors and keypoints
//            #pragma omp critical
//            {
//                allDescripts.push_back(descriptors);
//                keypoints.reserve(keypoints.size() + kpts.size());
//                keypoints.insert(keypoints.end(), kpts.begin(), kpts.end());
//
//            }
//
//        }
//    }
//    cout << "-> openCV SIFT descriptor computing spent " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " sec......" << endl;
//
//}
//
//void vl_visual_word_compute(Mat &allDescrip, Mat &kCenters) {
//    clock_t sTime = clock();
//    cout << "start k-means learning with "<<centers<<" centers..." << endl;
//    //all kmeans parameters setting corresponding with OpenCV setting
//    int dim = 128;
//    int numOfpts = allDescrip.rows;
//    double energy;
//    VlKMeans* km = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
//    vl_kmeans_set_algorithm(km,VlKMeansLloyd);
//
//    //data row major
//    float* data = allDescrip.ptr<float>(0);
//    vl_kmeans_init_centers_plus_plus(km, data, dim, numOfpts, centers);
//
//    vl_kmeans_set_num_repetitions(km,3);
//
//    /**The relative energy variation is calculated after the $t$ - th update
//    ** to the parameters as :
//    **
//    **  \[\epsilon_t = \frac{ E_{t - 1} -E_t }{E_0 - E_t} \]
//    **  set the relative energy variation to the initial minus current energy as the stop criteria
//    **  OpenCV uses the absolute variation of pixel corner position for kcenters as stop criteria, while VLFeat is the relative energy variation. 
//    **/
//    vl_kmeans_set_min_energy_variation(km, accuracy);
//    vl_kmeans_set_max_num_iterations(km, numOfItera);
//    energy = vl_kmeans_refine_centers(km, data, numOfpts);
//
//    //obtain kcenters result
//    const float* center_ptr = (float *)vl_kmeans_get_centers(km);
//
//    //iterate and stores the kcenters as cv::Mat
//    Mat1f centerDescrip(1,dim);
//    for (int i = 0; i < centers; i++) {
//        for (int j = 0; j < dim; j++) {
//            centerDescrip(0, j) = center_ptr[i * dim + j];
//        }
//        kCenters.push_back(centerDescrip);
//
//    }
//    cout << "-> VLFeat kmeans learning spent " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " sec" << endl;
//}
//
////compute the images descriptors and keypoints from the provided img paths
//void vlimg_descips_compute(std::vector<std::string>& paths, Mat &allDescripts, std::vector<KeyPoint> &cv_keypoints)
//{
//    size_t num_imgs = paths.size();
//    if (num_imgs == 0) {
//        throw std::invalid_argument("ERROR: provided imgs folder is empty!");
//    }
//    clock_t sTime = clock();
//    omp_set_num_threads(6);
//    #pragma omp parallel
//    {
//        #pragma omp for schedule(dynamic)
//        for (int i = 0; i < num_imgs; i = i + 1) {
//            // int minHessian = 400; // for SURF detector only
//            Mat grayImg;
//            Mat grayImgFl;
//            Mat1f imgDescrips;
//            if (!fs::exists(fs::path(paths[i]))) {
//                cout << "Warning: " << paths[i] << "does not exist!" << endl;
//                continue;
//            }
//            cvtColor(cv::imread(paths[i]), grayImg, COLOR_BGR2GRAY);
//
//            //surf to detect and compute
//            int width = grayImg.size().width;
//            int height = grayImg.size().height;
//
//
//            auto vl_sift = vl_sift_new(width, height, octave, noctaveLayer, 1);//define vl sift processor
//            vl_sift_set_edge_thresh(vl_sift, 10);
//            vl_sift_set_peak_thresh(vl_sift, 0.04);
//
//            grayImg.convertTo(grayImgFl,CV_32F, 1.0 / 255.0);
//           
//            float* img_ptr = grayImgFl.ptr<float>(0);
//
//            if (!grayImgFl.isContinuous()) {
//                std::cerr << "ERROR: when read " << paths[i] << " OpenCV finds uncontinuous address" << endl;
//                continue;
//            }
//
//            //go trough the loop of gaussian pyramid
//            int result = vl_sift_process_first_octave(vl_sift, img_ptr);
//
//            /*** define vlfeat pipeline
//             *  for each image calculate octaves
//             *  for each octave to calculate the scale spaces
//             *  for each scale space calculate keypoints
//             *  for each keypoints do thresholding and compute descriptors for each main orientation
//             *  add to the total statistics, descriptors and keypoints
//             ***/
//            while (result != VL_ERR_EOF) {
//                vl_sift_detect(vl_sift);
//
//                //to get each keypoints
//                int keyPtsNum = vl_sift_get_nkeypoints(vl_sift);
//
//                const auto* keypoints = vl_sift_get_keypoints(vl_sift);
//
//                //loop each keypoints and get orientation
//                for (int i = 0; i < keyPtsNum; i++) {
//                    double rot[4];
//                    int nOrit = vl_sift_calc_keypoint_orientations(vl_sift, rot, &keypoints[i]);
//
//                    //get the descriptors for each computed orientation in current image
//                    for (int j = 0; j < nOrit; j++) {
//                        float curr_descriptor[128];
//                        vl_sift_calc_keypoint_descriptor(vl_sift, curr_descriptor, &keypoints[i], rot[j]);
//                        Mat descripor(1, 128, CV_32F, curr_descriptor);
//                        imgDescrips.push_back(descripor);
//
//                        KeyPoint kpt(Point2f(keypoints[i].x, keypoints[i].y), keypoints[i].sigma, rot[j] * 180 / CV_PI, 0.f, keypoints[i].o);
//
//                        //push back keypoints in current keypoints
//                        #pragma omp critical
//                        {
//                            cv_keypoints.push_back(kpt);
//                        }
//                    }
//
//                }
//                result = vl_sift_process_next_octave(vl_sift);
//            }
//
//            //push back imge descriptors for current image
//            #pragma omp critical
//            {
//                allDescripts.push_back(imgDescrips);
//            }
//            //delete sift
//            vl_sift_delete(vl_sift);
//
//        }   
//    }
//    cout << "->vlfeat SIFT descriptor computing spent " << (clock() - sTime)/ double(CLOCKS_PER_SEC) << " sec......" << endl;
//}
//
////scan and read files for train and test img subfolder from the provided path
//cmdType readFiles(int argc, const char* argv[],std::vector<std::string>& trainFilePaths, std::vector<std::string> &testFilePaths) {
//    
//    //open files with link of images
//    CommandLineParser parser(argc, argv, keys);
//    cmdType cmd; //add arg options to this variables
//    
//    /*Mat img1 = imread(samples::findFile(parser.get<String>("input1"), false), IMREAD_GRAYSCALE);
//    Mat img2 = imread(samples::findFile(parser.get<String>("input2"), false), IMREAD_GRAYSCALE);*/
//    /**
//     * if path is provided, we compute the correspondence
//     *
//     * */
//    if (argc > 3) {
//        throw std::invalid_argument("provides too much arguments! please check -h for details");
//    }
//    /**if (argc == 2 && img1.empty() || img2.empty()) {
//        throw std::invalid_argument("please specify only either -path or -img!");
// 
//    }**/
//    if (parser.has("tool")) {
//        String tool = parser.get<String>("tool");
//        if (tool == "vlfeat") {
//            cmd.insert(std::make_pair<String, bool>("vlfeat", true));
//        }
//        else if(tool=="opencv")
//        {
//            cmd.insert(std::make_pair<String, bool>("opencv", true));
//        }
//        else if(tool=="both"){
//            cmd.insert(std::make_pair<String, bool>("both", true));
//        }
//        else {
//            throw std::invalid_argument("-tool= "+tool+" is not supported");
//        }
//    }
//
//    if(parser.has("path") && argc == 2){    //if (!imgs_path.empty() && img1.empty() && img2.empty() && img.empty()) {
//        fs::path imgs_path = parser.get<String>("path");
//        if (!imgs_path.empty() && !fs::exists(imgs_path)) {
//            cout << "ERROR: provided imgs path doesn't exist!" << endl;
//
//        }
//        cmd.insert(std::make_pair<String>("path", true));
//
//        //find the train imgs
//        fs::path train_path = imgs_path;
//        fs::path test_path = imgs_path;
//        train_path /= "train";
//        test_path /= "test";
//        cout << "Current training images path: " << train_path << endl;
//        cout << "Current testing images path: " << test_path << endl;
//        if (!fs::exists(train_path)) {
//            throw std::invalid_argument("train subfolder of provided path doesn't exist!");
//        }
//        else if (!fs::exists(train_path)) {
//            std::cerr << "WARNING: test subfolder for provided path doesn't exist! read training imgs from training path......" << endl;
//        }
//        else {
//            for (const auto& entry : fs::directory_iterator(test_path)) {
//                std::string::size_type idx;
//                idx = entry.path().string().rfind('.');
//                if (idx != std::string::npos)
//                {
//                    std::string extension = entry.path().string().substr(idx + 1);
//                    if (extension == "jpg") {
//                        testFilePaths.push_back(entry.path().string());
//                        cout << "test img is added and found at: " << entry.path().string() <<"......"<< endl;
//                    }
//                    else {
//                        cout << "img " + entry.path().string() + ": Extension" + extension + " is not supported dismiss the image" << endl;
//                    }
//                }
//            }
//        }
//
//        for (const auto& entry : fs::directory_iterator(train_path)) {
//            std::string::size_type idx;
//            idx = entry.path().string().rfind('.');
//            if (idx != std::string::npos)
//            {
//                std::string extension = entry.path().string().substr(idx + 1);
//                if (extension == "jpg") {
//                    trainFilePaths.push_back(entry.path().string());
//                    cout << "img is added and found at: " << entry.path().string() << "......"<<endl;
//                }
//                else {
//                    cout<<"img "+ entry.path().string()+": Extension"+extension+" is not supported dismiss the image"<<endl;
//                }
//            }
//        }
//        return cmd;
//    }
//    else if (parser.has("img") && argc == 2) {    //if (imgs_path.empty() && img1.empty() && img2.empty() && !img.empty()) {
//        fs::path img = parser.get<String>("img");
//        if (!fs::exists(img)) {
//            throw std::invalid_argument("provided -img path is not a valid image.");
//        }
//
//        cmd.insert(std::make_pair<String, bool>("img", true));
//
//        cout << "Current input image: " << img << endl;
//        std::string::size_type idx;
//        idx = img.string().rfind('.');
//        if (idx != std::string::npos)
//        {
//            std::string extension = img.string().substr(idx + 1);
//            if (extension == "jpg") {
//                trainFilePaths.push_back(img.string());
//                cout << "img is added and found at: " << img.string() << "......" << endl;
//            }
//            else {
//                throw std::invalid_argument("input img type " + extension + "is not supported");
//            }
//        }
//        return cmd;
//    }
//    else if(parser.has("input1") && parser.has("input2")&& argc == 3)//
//    {
//        fs::path img1 = parser.get<String>("input1");
//        fs::path img2 = parser.get<String>("input2");
//        if (!fs::exists(img1)|| !fs::exists(img2)) {
//            throw std::invalid_argument("provided -img1/-img2 path is not a valid image.");
//        }
//        cmd.insert(std::make_pair<String, bool>("inputs", true));
//
//        cout << "Two input image path is: \n ->img1: " << img1.string() << "\n ->img2: " << img2.string() << endl;
//        std::string::size_type idx1,idx2;
//        idx1 = img1.string().rfind('.');
//        idx2 = img2.string().rfind('.');
//        if (idx1 != std::string::npos && idx2 != std::string::npos)
//        {
//            std::string extension1 = img1.string().substr(idx1 + 1);
//            std::string extension2 = img2.string().substr(idx2 + 1);
//            if (extension1 == "jpg" && extension2 == "jpg") {
//                trainFilePaths.push_back(img1.string());
//                trainFilePaths.push_back(img2.string());
//            }
//            else {
//                throw std::invalid_argument("input img type "+extension1+"is not supported");
//            }
//        }
//        return cmd;
//    }
//    else {
//        throw std::invalid_argument("Confusing argument inputs please check your arguments.");
//    }
//
//    //TODO: function for read other types of arg inputs
//}
//
//
//void matchingtest(std::vector<String> &trainPath) {
//    std::vector<std::vector<KeyPoint>> ttl_kpts;
//    Mat1f ttl_descripts;
//    int dim = 128;
//    //get sift descriptor for two imgs
//    for (int i = 0; i < 2; i = i + 1) {
//        // int minHessian = 400; // for SURF detector only
//        Mat grayImg;
//        Mat grayImgFl;
//        //Mat1f imgDescrips;
//        std::vector<KeyPoint> cv_keypoints;
//        if (!fs::exists(fs::path(trainPath[i]))) {
//            cout << "Warning: " << trainPath[i] << "does not exist!" << endl;
//            continue;
//        }
//        cvtColor(cv::imread(trainPath[i]), grayImg, COLOR_BGR2GRAY);
//
//        //sift to detect and compute
//        int width = grayImg.size().width;
//        int height = grayImg.size().height;
//
//
//        auto vl_sift = vl_sift_new(width, height, octave, noctaveLayer, 1);//define vl sift processor
//        vl_sift_set_edge_thresh(vl_sift, 10);
//        vl_sift_set_peak_thresh(vl_sift, 0.04);
//
//        grayImg.convertTo(grayImgFl, CV_32F, 1.0 / 255.0);
//
//        float* img_ptr = grayImgFl.ptr<float>(0);
//
//        if (!grayImgFl.isContinuous()) {
//            std::cerr << "ERROR: when read " << trainPath[i] << " OpenCV finds uncontinuous address" << endl;
//            continue;
//        }
//
//        //go trough the loop of gaussian pyramid
//        int result = vl_sift_process_first_octave(vl_sift, img_ptr);
//
//        /*** define vlfeat pipeline
//         *  for each image calculate octaves
//         *  for each octave to calculate the scale spaces
//         *  for each scale space calculate keypoints
//         *  for each keypoints do thresholding and compute descriptors for each main orientation
//         *  add to the total statistics, descriptors and keypoints
//         ***/
//        //loop over octaves
//        while (result != VL_ERR_EOF) {
//            vl_sift_detect(vl_sift);
//
//            //to get each keypoints
//            int keyPtsNum = vl_sift_get_nkeypoints(vl_sift);
//
//            const auto* keypoints = vl_sift_get_keypoints(vl_sift);
//
//            //loop each keypoints and get orientation
//            for (int i = 0; i < keyPtsNum; i++) {
//                double rot[4];
//                int nOrit = vl_sift_calc_keypoint_orientations(vl_sift, rot, &keypoints[i]);
//
//                //get the descriptors for each computed orientation in current image
//                for (int j = 0; j < nOrit; j++) {
//                    float curr_descriptor[128];
//                    vl_sift_calc_keypoint_descriptor(vl_sift, curr_descriptor, &keypoints[i], rot[j]);
//                    Mat descripor(1, 128, CV_32F, curr_descriptor);
//                    ttl_descripts.push_back(descripor);
//
//                    KeyPoint kpt(Point2f(keypoints[i].x, keypoints[i].y), keypoints[i].sigma, rot[j] * 180 / CV_PI, 0.f, keypoints[i].o);
//
//                    //push back keypoints in current keypoints
//                    {
//                        cv_keypoints.push_back(kpt);
//                    }
//                }
//
//            }
//            result = vl_sift_process_next_octave(vl_sift);
//        }
//
//        //push back imge descriptors for current image
//        {
//            ttl_kpts.push_back(cv_keypoints);
//        }
//        //delete sift
//        vl_sift_delete(vl_sift);
//    }
//
//    //kd-tree building
//    
//    VlKDForest *tree =  vl_kdforest_new(VL_TYPE_FLOAT, dim, 1, VlDistanceL2);
//
//    //
//    //start building tree build from the first image
//    int numKpts1 = ttl_kpts[0].size();
//    int numKpts2 = ttl_kpts[1].size();
//
//    /*Mat1i NNs = Mat1i::zeros(numKpts2, numOfNN);
//    Mat1f NNdist = Mat1f::zeros(numKpts2, numOfNN);*/
//
//    if (ttl_descripts.isContinuous()) {
//        vl_kdforest_build(tree, numKpts1, ttl_descripts.ptr<float>(0));
//
//        //set searcher
//        vl_uint32* NNs = (vl_uint32*)vl_malloc(numOfNN*sizeof(vl_uint32) * numKpts2);
//        float* NNdist = (float*)vl_malloc(numOfNN*sizeof(float) * numKpts2);
//
//        VlKDForestSearcher* searcher = vl_kdforest_new_searcher(tree);
//        vl_kdforest_set_thresholding_method(tree, VL_KDTREE_MEDIAN); //use median as the criteria
//        VlKDForestNeighbor neighbors[2];
//        int numOfleaf = vl_kdforest_query_with_array(tree, NNs, 2, numKpts2, NNdist, ttl_descripts.ptr<float>(numKpts1));
//
//        cout << " -> total number of " << numOfleaf << " leafs are visited" << endl;
//        //draw matches
//        Mat img1 = imread(trainPath[0]), img2 = imread(trainPath[1]);
//        //build Dmatches
//        std::vector<DMatch> matches;
//        for (int i = 0; i < numKpts2; i++)
//        {
//            DMatch match = DMatch(i, NNs[numOfNN*i], NNdist[numOfNN*i]);
//            matches.push_back(match);
//        }
//        Mat outImg;
//        drawMatches(img2, ttl_kpts[1], img1, ttl_kpts[0], matches, outImg, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//        namedWindow("Matches", WINDOW_NORMAL);
//        imshow("Matches",outImg);
//        waitKey();
//    }
//    else {
//        cout << "ERROR: Matching stop for descriptors Mat is not continuous" << endl;
//    }
//
//
//}
//
//v
//#else
//int main()
//{
//    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
//    return 0;
//}
//#endif

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu


//find keypoints on the testimg and draw the keypoints graph
//bool testImg(std::vector<String> testPaths, Mat& kCenters) {
//    //use standard distance computation to find the NN
//    //only test the first img
//    if (testPaths.empty()) {
//        std::cout << "Test img path empty skip test img......" << endl;
//        return false;
//    }
//    Mat testDescrips;
//    testPaths.erase(testPaths.begin() + 1, testPaths.end());
//
//    std::vector<KeyPoint> cv_keypoints;
//    vlimg_descips_compute(testPaths, testDescrips, cv_keypoints);
//
//    //use OpenCV matching to get the labels
//    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
//    std::vector<DMatch> matchPairs;
//    matcher->match(testDescrips, kCenters, matchPairs);
//
//    Mat outImg, imgtest = imread(testPaths[0]);
//    cout << "Total " << cv_keypoints.size() << " keypoints found on the test image......" << endl;
//    //draw the testimg with opencv draw function
//
//    //write to file
//    Mat shadowCenters;
//    write_to_file("vltest_img", cv_keypoints, shadowCenters);
//    drawKeypoints(imgtest, cv_keypoints, outImg, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//    imshow("testOut", outImg);
//    imwrite("vlTest_result.jpg", outImg);
//    waitKey();
//    return true;
//}
