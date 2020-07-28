#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <omp.h>
#include "extractor.h"
#include "fileManager.h"
extern "C" {
    #include "vl/sift.h"
    #include "vl/generic.h"
    #include "vl/covdet.h"
}
using namespace std;
using namespace cv;
namespace fs = std::filesystem;
using params = fileManager::parameters;
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>
void extractor::openCVimg_descips_compute(std::vector<std::string>& paths, Mat& allDescripts, std::vector<KeyPoint>& keypoints)
{
    clock_t sTime = clock();
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
            Mat grayImg, purImg;
            Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();
            if (!fs::exists(fs::path(paths[i]))) {
                cout << "Warning: " << paths[i] << "does not exist, the image is ignored;" << endl;
                continue;
            }
            cvtColor(imread(paths[i]), grayImg, COLOR_BGR2GRAY);
            cv::resize(grayImg, purImg, cv::Size(), params::imgScale, params::imgScale, cv::INTER_AREA);
            grayImg = purImg;
            //surf to detect and compute
            Mat descriptors;
            std::vector<KeyPoint> kpts;
            detector->detectAndCompute(grayImg, noArray(), kpts, descriptors);

            //add to the total statistics, descriptors and keypoints
            #pragma omp critical
            {
                allDescripts.push_back(descriptors);
                keypoints.reserve(keypoints.size() + kpts.size());
                keypoints.insert(keypoints.end(), kpts.begin(), kpts.end());
            }
        }
    }
    cout << "   -> openCV SIFT descriptor computing spent " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " sec......" << endl;
}
#else
#error "Require OpenCV xfeature2d contrib_modules, please refer to opencv_contrib: https://github.com/opencv/opencv_contrib"
#endif
void extractor::vlimg_descips_compute(std::vector<std::string>& paths, Mat& allDescripts, std::vector<KeyPoint>& cv_keypoints)
{
    size_t num_imgs = paths.size();
    if (num_imgs == 0) {
        throw std::invalid_argument("ERROR: provided imgs folder is empty!");
    }
    clock_t sTime = clock();
    omp_set_num_threads(6);
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < num_imgs; i = i + 1) {
            // int minHessian = 400; // for SURF detector only
            Mat grayImg,ImgResize;
            Mat grayImgFl;
            Mat1f imgDescrips;
            std::vector<KeyPoint> imgKpts;
            if (!fs::exists(fs::path(paths[i]))) {
                cout << "vlfeat sift feature detection: Warning: " << paths[i] << " does not exist!" << endl;
                continue;
            }
            grayImg=cv::imread(paths[i],IMREAD_GRAYSCALE);

            //resize image
            cv::resize(grayImg, ImgResize, cv::Size(), params::imgScale, params::imgScale, cv::INTER_AREA);
            grayImg = ImgResize;
            //surf to detect and compute
            int width = grayImg.size().width;
            int height = grayImg.size().height;

            int noctave = params::octave;
            if (params::octave == -1) {
                noctave = log2(min(width, height));
            }
            auto vl_sift = vl_sift_new(width, height, noctave, params::noctaveLayer, params::firstOctaveInd);//define vl sift processor
            /*vl_sift_set_edge_thresh(vl_sift, params::siftEdgeThres);
            vl_sift_set_peak_thresh(vl_sift, params::siftPeakThres);*/

            /*grayImg.convertTo(grayImgFl, CV_32F, 1.0 / 255.0);*/
            grayImg.convertTo(grayImgFl, CV_32F);

            float* img_ptr = grayImgFl.ptr<float>(0);

            if (!grayImgFl.isContinuous()) {
                std::cerr << "ERROR: when read " << paths[i] << " OpenCV finds uncontinuous address" << endl;
                continue;
            }

            //go trough the loop of gaussian pyramid
            int result = vl_sift_process_first_octave(vl_sift, img_ptr);

            /*** define vlfeat pipeline
             *  for each image calculate octaves
             *  for each octave to calculate the scale spaces
             *  for each scale space calculate keypoints
             *  for each keypoints do thresholding and compute descriptors for each main orientation
             *  add to the total statistics, descriptors and keypoints
             ***/
            while (result != VL_ERR_EOF) {
                vl_sift_detect(vl_sift);

                //to get each keypoints
                int keyPtsNum = vl_sift_get_nkeypoints(vl_sift);

                const auto* keypoints = vl_sift_get_keypoints(vl_sift);

                //loop each keypoints and get orientation
                for (int i = 0; i < keyPtsNum; i++) {
                    double rot[4];
                    int nOrit = vl_sift_calc_keypoint_orientations(vl_sift, rot, &keypoints[i]);
                    //get the descriptors for each computed orientation in current image
                    for (int j = 0; j < nOrit; j++) {
                        float curr_descriptor[128];
                        vl_sift_calc_keypoint_descriptor(vl_sift, curr_descriptor, &keypoints[i], rot[j]);
                        Mat descript(1, 128, CV_32F, curr_descriptor);
                        imgDescrips.push_back(descript);
                        KeyPoint kpt(Point2f(keypoints[i].x, keypoints[i].y), keypoints[i].sigma, rot[j] * 180.0 / CV_PI, 0.f, keypoints[i].o);
                        
                        //push back keypoints in current keypoints
                        imgKpts.push_back(kpt);
                        
                    }

                }
                result = vl_sift_process_next_octave(vl_sift);
            }

            //push back imge descriptors and keypoints for current image
            #pragma omp critical
            {
                allDescripts.push_back(imgDescrips);
                cv_keypoints.insert(cv_keypoints.end(), imgKpts.begin(), imgKpts.end());
            }
            //delete sift
            vl_sift_delete(vl_sift);

        }
    }
   /* std::cout << "    -> total " << allDescripts.size() << " descriptors are created during the session" << endl;
    std::cout << "    ->vlfeat SIFT descriptor computing spent " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " sec......" << endl;*/
}

void extractor::vlimg_descips_compute_simple(Mat& img1, Mat& Descripts, std::vector<KeyPoint>& cv_keypoints) {
    //resize image
    auto sTime = clock();
    Mat ImgResize, ImgResizeF1;
    cv::resize(img1, ImgResize, cv::Size(), params::imgScale, params::imgScale, cv::INTER_AREA);

    //surf to detect and compute
    int width = ImgResize.size().width;
    int height = ImgResize.size().height;

    int noctave = params::octave;
    if (params::octave == -1) {
        noctave = log2(min(width, height));
    }
    auto vl_sift = vl_sift_new(width, height, noctave, params::noctaveLayer, params::firstOctaveInd);//define vl sift processor

    vl_sift_set_edge_thresh(vl_sift, params::siftEdgeThres);
    vl_sift_set_peak_thresh(vl_sift, params::siftPeakThres);

    ImgResize.convertTo(ImgResizeF1, CV_32F);

    float* img_ptr = ImgResizeF1.ptr<float>(0);

    if (!ImgResizeF1.isContinuous()) {
        throw std::invalid_argument("vlImg_descrip_extractor_simple: ERROR! incontinuous Mat object is found!");
    }

    //go trough the loop of gaussian pyramid
    int result = vl_sift_process_first_octave(vl_sift, img_ptr);

    /*** define vlfeat pipeline
     *  for each image calculate octaves
     *  for each octave to calculate the scale spaces
     *  for each scale space calculate keypoints
     *  for each keypoints do thresholding and compute descriptors for each main orientation
     *  add to the total statistics, descriptors and keypoints
     ***/
    while (result != VL_ERR_EOF) {
        vl_sift_detect(vl_sift);

        //to get each keypoints
        int keyPtsNum = vl_sift_get_nkeypoints(vl_sift);

        const auto* keypoints = vl_sift_get_keypoints(vl_sift);

        //loop each keypoints and get orientation
        for (int i = 0; i < keyPtsNum; i++) {
            double rot[4];
            int nOrit = vl_sift_calc_keypoint_orientations(vl_sift, rot, &keypoints[i]);

            //get the descriptors for each computed orientation in current image
            for (int j = 0; j < nOrit; j++) {
                float curr_descriptor[128];
                vl_sift_calc_keypoint_descriptor(vl_sift, curr_descriptor, &keypoints[i], rot[j]);
                Mat descript(1, 128, CV_32F, curr_descriptor);
                Descripts.push_back(descript);

                KeyPoint kpt(Point2f(keypoints[i].x, keypoints[i].y), keypoints[i].sigma, rot[j] * 180.0 / CV_PI, 0.f, keypoints[i].o);

                //push back keypoints in current keypoints
                cv_keypoints.push_back(kpt);
            }

        }
        result = vl_sift_process_next_octave(vl_sift);
    }
    //delete sift
    vl_sift_delete(vl_sift);
    /*cout << "    -> total " << Descripts.size() << " descriptors are created during the session" << endl;
    cout << "    ->vlfeat SIFT descriptor computing spent " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " sec......" << endl;*/
}

void extractor::covdetSIFT(cv::Mat &img) {
    size_t numRows = img.rows;
    size_t numCols = img.cols;
    
    //set covdet detector to different of gaussian
    VlCovDet* covdet = vl_covdet_new(VlCovDetMethod::VL_COVDET_METHOD_DOG);

    // set various parameters
    vl_covdet_set_first_octave(covdet, -1); //covdet default = -1
    // vl_covdet_set_num_octaves(covdet, -1); //covdet default = -1
    // vl_covdet_set_max_num_orientations(covdet, 4);//covdet default = 4
    // vl_covdet_set_non_extrema_suppression_threshold(covdet, 0.5); //covdet default = 0.5
    vl_covdet_set_octave_resolution(covdet, params::noctaveLayer); //
    vl_covdet_set_peak_threshold(covdet, params::siftPeakThres); //covdet default = 0.01
    vl_covdet_set_edge_threshold(covdet, params::siftEdgeThres); //covdet default = 10.0

    //process image
    cv::Mat imgFloat;
    img.convertTo(imgFloat, CV_32F);
    vl_covdet_put_image(covdet,imgFloat.ptr<float>(0) , numRows, numCols);

    //do detection
    vl_covdet_detect(covdet);

    //drop marginal features use recommended value 1
    vl_covdet_drop_features_outside(covdet, 1);

    // compute the affine shape of the features, drop feature that cannot produce reliable affine shape
    vl_covdet_extract_affine_shape(covdet);

    // compute the orientation of the features, maximum maxNumOrient orientation are created and feature are duplicated
    vl_covdet_extract_orientations(covdet);

    //get feature frame back
    vl_size numFeatures = vl_covdet_get_num_features(covdet);
    VlCovDetFeature const* feature = (VlCovDetFeature const* )vl_covdet_get_features(covdet);

    //process the features sift process the patch around 16*16 windows around the feature
    VlSiftFilt* sift = vl_sift_new(16, 16, 1, 3, 0);

    //use recommended setting
    vl_size dim = 128;
    vl_size patchResolution = 15;
    double patchRelativeExtent = 7.5;
    double patchRelativeSmoothing = 1;
    vl_size w = 2 * patchResolution + 1;
    double patchStep = patchRelativeExtent / patchResolution;
    //construct SIFT features
    //we store the keypoints and descriptors in form of opencv
    std::vector<KeyPoint> kpts;
    cv::Mat descrips(numFeatures,dim,CV_32F);
    for (int i = 0; i < numFeatures; i++) {
        //extract patches from frames
        std::vector<float> patch(w * w);
        std::vector<float> grads(2 * w * w);
        vl_covdet_extract_patch_for_frame(covdet, patch.data(), 
                                          patchResolution, patchRelativeExtent, 
                                          patchRelativeSmoothing, feature[i].frame);
        //computes and stores amplitude and angle gradient in the same grads vector
        vl_imgradient_polar_f(&grads[0], &grads[1], 2, 2 * w, patch.data(), w, w, w);

        vl_sift_calc_raw_descriptor(sift, grads.data(),
            descrips.ptr<float>(i), (int)w, (int)w,
            (double)(w - 1) / 2, (double)(w - 1) / 2,
            (double)patchRelativeExtent / (3.0 * (4 + 1) / 2) / patchStep,
            VL_PI / 2); // kpt scale: because img shrink for 1 / patchStep, and smoothing is 1.0; Besides, keypoint should at the center of extracted patch
        //quote: In order to be equivalent to a standard SIFT descriptor the image gradient must be computed at a smoothing level equal to the scale of the keypoint
        
    }
    

}



