// vsImgProject.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
extern "C" {
    //#include //"vl/vlad.h"
    #include "vl/sift.h"
    #include "vl/generic.h"
    #include "vl/kmeans.h"
}
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/persistence.hpp>
#include "Shlwapi.h"
#include "omp.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <string>
#include <iostream>
using namespace cv;
namespace fs = std::filesystem;
using std::cout;
using std::endl;
const char *keys =
     "{ help h |                  | Print help message. }"
     "{ path   |                  | Path to the image folder, not comp. with input1/2 }"
     "{ input1 |                  | Path to input image 1, not comp. with path }"
     "{ input2 |                  | Path to input image 2, not comp. with path }";

int octave = 8;       // number of octave used in the sift detection
int noctaveLayer = 3; // scale layers per octave
int octave_start = 1; // learning start from 1th octave, -1 for more details
double sigma_0 = 1.6; // sigma for the #0 octave
int centers = 200;    // k-means center detection, defines the number of centers

//k-means learning the cluters from the provided descriptors
void visual_word_comp(Mat &allDescrip, Mat &kCenters) {
    clock_t sTime = clock();
    cout << "start k-means learning..." << endl;
    //all kmeans parameters setting corresponding with OpenCV setting
    int dim = 128;
    int numOfpts = allDescrip.rows;
    double energy;
    VlKMeans* km = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
    vl_kmeans_set_algorithm(km,VlKMeansLloyd);

    //data row major
    float* data = allDescrip.ptr<float>(0);
    vl_kmeans_init_centers_plus_plus(km, data, dim, numOfpts, centers);

    vl_kmeans_set_num_repetitions(km,3);

    /**The relative energy variation is calculated after the $t$ - th update
    ** to the parameters as :
    **
    **  \[\epsilon_t = \frac{ E_{t - 1} -E_t }{E_0 - E_t} \]
    **  set the relative energy variation to the initial minus current energy as the stop criteria
    **  OpenCV uses the absolute variation of pixel corner position for kcenters as stop criteria, while VLFeat is the relative energy variation. 
    **/
    vl_kmeans_set_min_energy_variation(km, 1e-3);
    vl_kmeans_set_max_num_iterations(km, 20);

    energy = vl_kmeans_refine_centers(km, data, numOfpts);

    //obtain kcenters result
    const float* center_ptr = (float *)vl_kmeans_get_centers(km);

    //iterate and stores the kcenters as cv::Mat
    Mat1f centerDescrip(1,dim);
    for (int i = 0; i < centers; i++) {
        for (int j = 0; j < dim; j++) {
            centerDescrip(0, j) = center_ptr[i * dim + j];
        }
        kCenters.push_back(centerDescrip);

    }
    cout << "-> kmeans learning spent " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " sec" << endl;
}

//compute the images descriptors and keypoints from the provided img paths
void vlimg_descips_compute(std::vector<std::string>& paths, Mat &allDescripts, std::vector<KeyPoint> &cv_keypoints)
{
    size_t num_imgs = paths.size();
    if (num_imgs == 0) {
        throw std::invalid_argument("imgs folder is empty!");
    }
    omp_set_num_threads(6);
    
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < num_imgs; i = i + 1) {
            // int minHessian = 400; // for SURF detector only
            Mat grayImg;
            Mat grayImgFl;
            Mat1f imgDescrips;
            if (!fs::exists(fs::path(paths[i]))) {
                cout << "Warning: " << paths[i] << "does not exist, pass the image" << endl;
                continue;
            }
            cvtColor(cv::imread(paths[i]), grayImg, COLOR_BGR2GRAY);

            //surf to detect and compute
            int width = grayImg.size().width;
            int height = grayImg.size().height;


            auto vl_sift = vl_sift_new(width, height, octave, noctaveLayer, 1);//define vl sift processor
            vl_sift_set_edge_thresh(vl_sift, 10);
            vl_sift_set_peak_thresh(vl_sift, 0.04);

            grayImg.convertTo(grayImgFl,CV_32F, 1.0 / 255.0);
           
            float* img_ptr = grayImgFl.ptr<float>(0);

            if (!grayImgFl.isContinuous()) {
                std::cerr << "ERROR: when read " << paths[i] << " OpenCV finds uncontinuous address" << endl;
                continue;
            }

            //go trough the loop of gaussian pyramid
            int result = vl_sift_process_first_octave(vl_sift, img_ptr);
            while (result != VL_ERR_EOF) {
                vl_sift_detect(vl_sift);

                //to get each keypoints
                int keyPtsNum = vl_sift_get_nkeypoints(vl_sift);

                const auto* keypoints = vl_sift_get_keypoints(vl_sift);

                //loop each keypoints and get orientation
                for (int i = 0; i < keyPtsNum; i++) {
                    double rot[4];
                    int nOrit = vl_sift_calc_keypoint_orientations(vl_sift, rot, &keypoints[i]);

                    //get the descriptors for each computed orientation
                    for (int j = 0; j < nOrit; j++) {
                        float curr_descriptor[128];
                        vl_sift_calc_keypoint_descriptor(vl_sift, curr_descriptor, &keypoints[i], rot[j]);
                        Mat descripor(1, 128, CV_32F, curr_descriptor);
                        imgDescrips.push_back(descripor);

                        KeyPoint kpt(Point2f(keypoints[i].x, keypoints[i].y), keypoints[i].sigma, rot[j] * 180 / CV_PI, 0.f, keypoints[i].o);
                        #pragma omp critical
                        {
                            cv_keypoints.push_back(kpt);
                        }
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
                allDescripts.push_back(imgDescrips);
            }
            //delete sift
            vl_sift_delete(vl_sift);

        }
    }

}

//scan and read files for train and test img subfolder from the provided path
bool readFiles(int argc, const char* argv[],std::vector<std::string>& trainFilePaths, std::vector<std::string> &testFilePaths) {
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

    //TODO: function for read other types of arg inputs
}

//find keypoints on the testimg and draw the keypoints graph
bool testImg(std::vector<String> testPaths, Mat& kCenters) {
    //use standard distance computation to find the NN
    //only test the first img
    if (testPaths.empty()) {
        std::cout << "Test img path empty skip test img......" << endl;
        return false;
    }
    Mat testDescrips;
    testPaths.erase(testPaths.begin() + 1, testPaths.end());
    
    std::vector<KeyPoint> cv_keypoints;
    vlimg_descips_compute(testPaths, testDescrips,cv_keypoints);

    //use OpenCV matching to get the labels
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    std::vector<DMatch> matchPairs;
    matcher->match(testDescrips, kCenters, matchPairs);
    
    Mat outImg, imgtest = imread(testPaths[0]);
    cout << "Total " << cv_keypoints.size() << " keypoints found on the test image......" << endl;
    //draw the testimg with opencv draw function
    drawKeypoints(imgtest,cv_keypoints,outImg,Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("testOut", outImg);
    waitKey();
    return true;
}

int main(int argc, const char* argv[]) {
    VL_PRINT("!------- VLFeat detection program starts! ------!\n");
    std::vector<std::string> trainPaths, testPaths;
    if(!readFiles(argc, argv, trainPaths, testPaths)) //fed train imgs and test imgs path to the corresponding string vectors.
        return 0;
    Mat allDescrips,kCenters;
    std::vector<KeyPoint> keypoints;
    cout << "total files found: " << trainPaths.size() << endl;
    try {
        vlimg_descips_compute(trainPaths, allDescrips,keypoints);
    }
    catch (std::invalid_argument& e) {
        cout << e.what() << endl;
    };

    //free memory
    std::vector<KeyPoint>().swap(keypoints);

    //cout << "debug info about allDescrip: " << allDescrips << endl;
    //train k-means classifier
    visual_word_comp(allDescrips, kCenters);

    //print out and validate the kcenters
    cout << "The computed result of kCenters are: " << kCenters.rows << ", " << kCenters.cols << "......."<<endl;
    cout << "Start testing on new image......" << endl;

    testImg(testPaths, kCenters);
    VL_PRINT("!------- This program stop here ------!\n");
    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu