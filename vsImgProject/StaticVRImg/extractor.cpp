
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <set>
#include <boost/filesystem.hpp>
#include <omp.h>
#include "extractor.h"
#include "fileManager.h"
#include "helper.h"

//#include "util/misc.h"
//#include "util/opengl_utils.h"
//#include <SiftGPU/SiftGPU.h>
//#include <feature/sift.h>

extern "C" {
    #include "vl/sift.h"
    #include "vl/generic.h"
    #include "vl/covdet.h"
}
using namespace std;
using namespace cv;
namespace fs = boost::filesystem;
using params = fileManager::parameters;
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>

//change features to other form ---- from colmap
cv::Mat TransformVLFeatToUBCFeatureDescriptors(
    const Mat& vlfeat_descriptors) {
    Mat ubc_descriptors(vlfeat_descriptors.rows, vlfeat_descriptors.cols,CV_8U);
    const std::array<int, 8> q{ {0, 7, 6, 5, 4, 3, 2, 1} };
    for (int n = 0; n < vlfeat_descriptors.rows; ++n) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 8; ++k) {
                    ubc_descriptors.at<uint8_t>(n, 8 * (j + 4 * i) + q[k]) =
                        vlfeat_descriptors.at<uint8_t>(n, 8 * (j + 4 * i) + k);
                }
            }
        }
    }
    return ubc_descriptors;
}
void vl_feature_scale_extraction(VlCovDet* self) {
    vl_index i, j;
    vl_bool dropFeaturesWithoutScale = VL_TRUE;
    vl_size numFeatures = vl_covdet_get_num_features(self);
    VlCovDetFeature* features = (VlCovDetFeature * )vl_covdet_get_features(self);

    for (i = 0; i < (signed)numFeatures; ++i) {
        vl_size numScales;
        VlCovDetFeature feature = features[i];
        VlCovDetFeatureLaplacianScale const* scales =
            vl_covdet_extract_laplacian_scales_for_frame(self, &numScales, feature.frame);


        if (numScales == 0 && dropFeaturesWithoutScale) {
            features[i].peakScore = 0;
            continue;
        }
        vl_index bestScale=0;
        double chosenScale = 0.0;

        //choose the largest possible scale
        for (j = 0; j < (signed)numScales; ++j) {
            if (scales[j].scale > chosenScale) {
                chosenScale = scales[j].scale;
                bestScale = j;
            }
        }
        VlCovDetFeature* scaled = &features[i];
        scaled->laplacianScaleScore = scales[bestScale].score;
        scaled->frame.a11 *= scales[bestScale].scale;
        scaled->frame.a21 *= scales[bestScale].scale;
        scaled->frame.a12 *= scales[bestScale].scale;
        scaled->frame.a22 *= scales[bestScale].scale;
    }

}
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
            Mat grayImg,ImgResize;
            Mat grayImgFl;
            Mat imgDescrips,truntimgDescrips;
            std::vector<KeyPoint> imgKpts,truntimgKpts;
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
            vl_sift_set_edge_thresh(vl_sift, params::siftEdgeThres);
            vl_sift_set_peak_thresh(vl_sift, params::siftPeakThres);

            grayImg.convertTo(grayImgFl, CV_32F, 1.0f / 255.0f);
            /*grayImg.convertTo(grayImgFl, CV_32F);*/

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
            //build temporal keypoints containers for each levels
            std::vector<int> keyPointsNums;
            int levels_index = -1;
            while (!result) {
                
                vl_sift_detect(vl_sift);

                //to get each keypoints
                int keyPtsNum = vl_sift_get_nkeypoints(vl_sift);

                const auto* keypoints = vl_sift_get_keypoints(vl_sift);
                if (keyPtsNum == 0) {
                    result = vl_sift_process_next_octave(vl_sift); //zero keypoints loop to next octave layers
                    continue;
                }
                int current_levels = -1;
                //loop each keypoints and get orientation with limitation: based on the octave levels check 
                for (int i = 0; i < keyPtsNum; i++) {
                    if (current_levels != keypoints[i].is) {
                        keyPointsNums.push_back(0);
                        current_levels = keypoints[i].is;
                        levels_index++;
                    }
                    keyPointsNums.back()++; //increment keypoint
                    double rot[4];
                    int nOrit = vl_sift_calc_keypoint_orientations(vl_sift, rot, &keypoints[i]);
                    const int numOrient = std::min(params::maxNumOrient, nOrit);
                    //get the descriptors for each computed orientation in current image
                    for (int j = 0; j < numOrient; j++) {
                        float curr_descriptor[128];
                        vl_sift_calc_keypoint_descriptor(vl_sift, curr_descriptor, &keypoints[i], rot[j]);
                        Mat descript(1, 128, CV_32F, curr_descriptor);
                        
                        //normalizing the descriptors
                        normalize(descript, descript, 1,0,NORM_L2);
                        //transform descrip back to uint8_t
                        cv::Mat uintDescript = helper::DescriptorFloatToUint(descript);

                        imgDescrips.push_back(descript);
                        //use class_id as the level index
                        KeyPoint kpt(Point2f(keypoints[i].x+0.5f, keypoints[i].y+0.5f), keypoints[i].sigma, rot[j] * 180.0 / CV_PI, 0.f, keypoints[i].o,levels_index);
                        
                        //push back keypoints in current keypoints
                        imgKpts.push_back(kpt);
                        
                    }

                }
                result = vl_sift_process_next_octave(vl_sift);
            }

            //iterate through the keypoints container and check the num limits

            int numKpts = 0;
            int levelToKeep=0;
            for (int i = keyPointsNums.size() - 1; i >= 0; i--) {
                numKpts += keyPointsNums[i];
                if (numKpts > params::maxNumFeatures) {
                    levelToKeep = i;
                    break;
                }
            }

            //reserve enough space for the final container
            truntimgKpts.reserve(numKpts*params::maxNumOrient);
            truntimgDescrips.reserve(numKpts* params::maxNumOrient);
            //iterate and add kpts and descripts to the container
            for(int k = imgKpts.size() - 1; k >= 0; k--) {
                if (imgKpts[k].class_id >= levelToKeep) {
                    truntimgKpts.push_back(imgKpts[k]);
                    truntimgDescrips.push_back(imgDescrips.row(k));
                }
                else
                {
                    break;
                }
            }
            truntimgDescrips = TransformVLFeatToUBCFeatureDescriptors(truntimgDescrips);
            //push back imge descriptors and keypoints for current image
            #pragma omp critical
            {
                allDescripts.push_back(truntimgDescrips);
                cv_keypoints.insert(cv_keypoints.end(), truntimgKpts.begin(), truntimgKpts.end());
            }
            //delete sift
            vl_sift_delete(vl_sift);

        }
    }
   /* std::cout << "    -> total " << allDescripts.size() << " descriptors are created during the session" << endl;
    std::cout << "    ->vlfeat SIFT descriptor computing spent " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " sec......" << endl;*/
}

void extractor::vlimg_descips_compute_simple(Mat& img1, Mat& Descripts, std::vector<KeyPoint>& cv_keypoints, colmap::Bitmap *bitmap) {
    //resize image
    auto sTime = clock();
    int width = 0, height = 0;
    std::vector<KeyPoint> imgKpts, truntimgKpts;
    Mat imgDescrips, truntimgDescrips;
    int noctave = params::octave;
    float* img_ptr = nullptr;
    std::vector<float> data_float;
    Mat grayImg;
    grayImg = img1;
    Mat grayImgFl(grayImg.rows, grayImg.cols, CV_32F);
    
    if (!bitmap) {
        //resize image
        cv::resize(grayImg, grayImg, cv::Size(), params::imgScale, params::imgScale, cv::INTER_AREA);

        //surf to detect and compute
        width = grayImg.size().width;
        height = grayImg.size().height;

        if (params::octave == -1) {
            noctave = log2(min(width, height));
        }
      
        grayImg.convertTo(grayImgFl, CV_32F, 1.0f / 255.0f);
        /*grayImg.convertTo(grayImgFl, CV_32F);*/

        if (!grayImgFl.isContinuous()) {
            std::cerr << "ERROR: when read img in vlimg_descript_compute_simple OpenCV finds uncontinuous address" << endl;
            return;
        }

        img_ptr = grayImgFl.ptr<float>(0);
    }
    else
    {
        width = bitmap->Width();
        height = bitmap->Height();
        const std::vector<uint8_t> data_uint8 = bitmap->ConvertToRowMajorArray();
        data_float.resize(data_uint8.size());
        for (size_t i = 0; i < data_uint8.size(); ++i) {
            data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
        }
        img_ptr = data_float.data();
    }

    
    auto vl_sift = vl_sift_new(width, height, noctave, params::noctaveLayer, params::firstOctaveInd);//define vl sift processor
    vl_sift_set_edge_thresh(vl_sift, params::siftEdgeThres);
    vl_sift_set_peak_thresh(vl_sift, params::siftPeakThres);

    //go trough the loop of gaussian pyramid
    int result = vl_sift_process_first_octave(vl_sift, img_ptr);

    /*** define vlfeat pipeline
     *  for each image calculate octaves
     *  for each octave to calculate the scale spaces
     *  for each scale space calculate keypoints
     *  for each keypoints do thresholding and compute descriptors for each main orientation
     *  add to the total statistics, descriptors and keypoints
     ***/
     //build temporal keypoints containers for each levels
    std::vector<int> keyPointsNums;
    int levels_index = -1;
    while (!result) {

        vl_sift_detect(vl_sift);

        //to get each keypoints
        int keyPtsNum = vl_sift_get_nkeypoints(vl_sift);

        const auto* keypoints = vl_sift_get_keypoints(vl_sift);
        if (keyPtsNum == 0) {
            result = vl_sift_process_next_octave(vl_sift); //zero keypoint obatained than loop to next octave layers
            continue;
        }
        int current_levels = -1;
        //loop each keypoints and get orientation with limitation: based on the octave levels check 
        for (int i = 0; i < keyPtsNum; i++) {
            if (current_levels != keypoints[i].is) {
                keyPointsNums.push_back(0);
                current_levels = keypoints[i].is;
                levels_index++;
            }
            keyPointsNums.back()++; //increment keypoint
            double rot[4];
            int nOrit = vl_sift_calc_keypoint_orientations(vl_sift, rot, &keypoints[i]);
            const int numOrient = std::min(params::maxNumOrient, nOrit);
            //get the descriptors for each computed orientation in current image
            for (int j = 0; j < numOrient; j++) {
                float curr_descriptor[128];
                vl_sift_calc_keypoint_descriptor(vl_sift, curr_descriptor, &keypoints[i], rot[j]);
                Mat descript(1, 128, CV_32F, curr_descriptor);

                //normalizing the descriptors with L1Root method/TODO: with L2 normalization
                normalize(descript, descript, 1, 0, NORM_L1);
                cv::sqrt(descript, descript);

                //transform descrip back to uint8_t
                cv::Mat uintDescript = helper::DescriptorFloatToUint(descript);

                imgDescrips.push_back(uintDescript);
                //use class_id as the level index
                KeyPoint kpt(Point2f(keypoints[i].x + 0.5f, keypoints[i].y + 0.5f), keypoints[i].sigma, rot[j] * 180.0 / CV_PI, 0.f, keypoints[i].o, levels_index);

                //push back keypoints in current keypoints
                imgKpts.push_back(kpt);

            }

        }
        result = vl_sift_process_next_octave(vl_sift);
    }

    //iterate through the keypoints container and check the num limits
    int numKpts = 0;
    int levelToKeep = 0;
    for (int i = keyPointsNums.size() - 1; i >= 0; i--) {
        numKpts += keyPointsNums[i];
        if (numKpts > params::maxNumFeatures) {
            levelToKeep = i;
            break;
        }
    }

    //reserve enough space for the final container
    Descripts.reserve(numKpts * params::maxNumOrient);
    cv_keypoints.reserve(numKpts * params::maxNumOrient);
    //iterate and add kpts and descripts to the container
    for (int k = 0; k < imgKpts.size(); k++) {
        if (imgKpts[k].class_id >= levelToKeep) {
            cv_keypoints.push_back(imgKpts[k]);
            Descripts.push_back(imgDescrips.row(k));
        }
        else
        {
            continue;
        }
    }
    Descripts = TransformVLFeatToUBCFeatureDescriptors(Descripts);
    //push back imge descriptors and keypoints for current image
    //delete sift
    vl_sift_delete(vl_sift);
}

void extractor::covdetSIFT(cv::Mat &img, Mat& descriptors, std::vector<KeyPoint> &kpts) {
    if (!img.data) {
        std::cout << "covdetSIFT: ERROR: empty img!\n";
        return;
    }
    descriptors.release();
    cv::Mat imgResize;
    cv::resize(img, imgResize, cv::Size(), params::imgScale, params::imgScale, cv::INTER_AREA);
    size_t width = imgResize.size().width;
    size_t height = imgResize.size().height;
    int noctave;
    kpts.clear();

    //set mapper container that for descriptor selector
    std::vector<float> responseScores;

    //set covdet detector to DoG method(SIFT)
    VlCovDet* covdet = vl_covdet_new(VlCovDetMethod::VL_COVDET_METHOD_DOG);

    //use maximum num of octave
    if (params::octave == -1) {
        noctave = log2(min(width, height));
    }
    else
        noctave = params::octave;
    vl_covdet_set_num_octaves(covdet, noctave);

    // set various parameters
    vl_covdet_set_first_octave(covdet, params::firstOctaveInd); //covdet default = -1
    // vl_covdet_set_num_octaves(covdet, -1); //covdet default = -1
    vl_covdet_set_max_num_orientations(covdet, 2);//covdet default = 4
    // vl_covdet_set_non_extrema_suppression_threshold(covdet, 0.5); //covdet default = 0.5
    vl_covdet_set_octave_resolution(covdet, params::noctaveLayer); //
    vl_covdet_set_peak_threshold(covdet, params::siftPeakThres); //covdet default = 0.01
    vl_covdet_set_edge_threshold(covdet, params::siftEdgeThres); //covdet default = 10.0

    //process image and input to detector
    cv::Mat imgFloat;
    imgResize.convertTo(imgFloat, CV_32F);
    if (!imgFloat.isContinuous()) {
        throw std::invalid_argument("covdet: ERROR: incontinous Mat is found!");
    }
    vl_covdet_put_image(covdet,imgFloat.ptr<float>(0) , width, height);

    //do detection on image
    vl_covdet_detect(covdet);

    //drop marginal features use recommended value 1
    vl_covdet_drop_features_outside(covdet, 1);

    // compute the affine shape of the features, drop feature that cannot produce reliable affine shape
    /*vl_covdet_extract_affine_shape(covdet);*/

    // compute the orientation of the features, maximum num(maxNumOrient) orientation are created and corresponding feature are duplicated
    vl_covdet_extract_orientations(covdet);

    // customized extract scales function
    vl_feature_scale_extraction(covdet);
    /*vl_covdet_extract_laplacian_scales(covdet);*/

    //get feature frame back
    vl_size numFeatures = vl_covdet_get_num_features(covdet);
    VlCovDetFeature const* feature = (VlCovDetFeature const*)vl_covdet_get_features(covdet);

    //init sift feature builder process the patch 16*16 windows around the feature
    VlSiftFilt* sift = vl_sift_new(16, 16, 1, 3, 0);

    //use recommended setting
    vl_size dim = 128;
    vl_size patchResolution = 15;
    double patchRelativeExtent = 7.5;
    double patchRelativeSmoothing = 1;
    vl_size w = 2 * patchResolution + 1;
    double patchStep = patchRelativeExtent / patchResolution;

    /***construct SIFT features***/
    //we store the keypoints and descriptors in form of opencv
    cv::Mat descrips;
    descrips.reserve(2000);
    std::vector<KeyPoint> tempKpts;
    for (int i = 0; i < numFeatures; i++) {
        auto currentFrame = feature[i].frame;
        //build keypoint

        KeyPoint kpt;
        kpt.pt.x = currentFrame.x;
        kpt.pt.y = currentFrame.y;
        kpt.response = feature[i].peakScore;
        if (!kpt.response) {
            continue;
        }
        responseScores.push_back(kpt.response);
        //delete zero scale feature
        
        //extract orientation
        //vl_size numOrient;
        //VlCovDetFeatureOrientation* orients = vl_covdet_extract_orientations_for_frame(covdet, &numOrient, currentFrame);
        ////after vl_covdet_extract_orientations(), it should always one orientation
        //kpt.angle = orients[0].angle * 180.0 / CV_PI; //opencv use degree instead

        kpt.angle = atan2(currentFrame.a21, currentFrame.a11) * 180.0f / VL_PI;

        //extract scale
        //vl_size numScales;
        //VlCovDetFeatureLaplacianScale* scales = vl_covdet_extract_laplacian_scales_for_frame(covdet, &numScales, currentFrame);
        ////By vl_sift, always keep the largest possible scale
        //double scale = 0.0;
        //for (int i = 0; i < numScales; i++) {
        //    if (scales[i].scale > scale)
        //        scale = scales[i].scale;
        //}

        float det = currentFrame.a11 * currentFrame.a22 - currentFrame.a12 * currentFrame.a21;
        float size = sqrt(fabs(det));
        kpt.size = size;

        //pushback
        tempKpts.push_back(kpt);
        
        //extract patch from frame and build sift descriptor
        std::vector<float> patch(w * w);
        std::vector<float> grads(2 * w * w);
        vl_covdet_extract_patch_for_frame(covdet, patch.data(), 
                                          patchResolution, patchRelativeExtent, 
                                          patchRelativeSmoothing, currentFrame);
        //computes and stores amplitude and angle gradient in the grads vector with two layers.
        vl_imgradient_polar_f(&grads[0], &grads[1], 2, 2 * w, patch.data(), w, w, w);
        Mat currentDescrip(1, dim, CV_32F);
        vl_sift_calc_raw_descriptor(sift, grads.data(),
            currentDescrip.ptr<float>(0), (int)w, (int)w,
            (double)(w - 1) / 2, (double)(w - 1) / 2,
            (double)patchRelativeExtent / (3.0 * (4 + 1) / 2) / patchStep,
            VL_PI / 2); // scale seting: because img shrink for 1 / patchStep, and inner smoothing is 1.0; Besides, keypoint should at the center of extracted patch
        //quote: In order to be equivalent to a standard SIFT descriptor the image gradient must be computed at a smoothing level equal to the scale of the keypoint
        /*cv::Mat tempDesript(1, 128, CV_32F);
        flip_descriptor(tempDesript, currentDescrip.ptr<float>(0));*/
        descrips.push_back(currentDescrip);
    }
    
    //sort and only stores the keypoints/descriptors with the highest response. From highest to smallest
    if (params::maxNumFeatures != -1) {
        descriptors.reserve(params::maxNumFeatures);
        std::vector<int> response_index(responseScores.size());
        std::iota(response_index.begin(), response_index.end(), 0);
        std::sort(response_index.begin(), response_index.end(), [&](size_t left, size_t right) {return fabs(responseScores[left]) > fabs(responseScores[right]); });
        for (int i = 0; i < params::maxNumFeatures && i<responseScores.size(); i++) {
            //extract descriptors and keypoints
            descriptors.push_back(descrips.row(response_index[i]));
            kpts.push_back(tempKpts[response_index[i]]);
        }
    }
    else {
        descriptors = descrips.clone();
    }
    //
    vl_sift_delete(sift);
    vl_covdet_delete(covdet);
}

//void extractor::siftGPU_descips_compute_simple(std::vector<colmap::Bitmap> queryImgs, std::vector<colmap::FeatureKeypoints>& kpts, std::vector<colmap::FeatureDescriptors>& descripts) {
//    colmap::SiftExtractionOptions sift_options_;
//    sift_options_.max_num_features = fileManager::parameters::maxNumFeatures;
//    sift_options_.max_num_orientations = fileManager::parameters::maxNumOrient;
//    std::unique_ptr<colmap::OpenGLContextManager> opengl_context_;
//    CHECK(opengl_context_);
//    opengl_context_->MakeCurrent();
//    std::vector<int> gpu_indices = colmap::CSVToVector<int>(sift_options_.gpu_index);
//    CHECK_GT(gpu_indices.size(), 0);
//    
//    auto sift_gpu_options = sift_options_;
//    auto& gpu_index = gpu_indices[0];
//    sift_gpu_options.gpu_index = std::to_string(gpu_index);
//
//    kpts.clear();
//    kpts.resize(queryImgs.size());
//    descripts.clear();
//    descripts.resize(queryImgs.size());
//    std::unique_ptr<SiftGPU> sift_gpu;
//    if (sift_options_.use_gpu) {
//        sift_gpu.reset(new SiftGPU);
//        if (!CreateSiftGPUExtractor(sift_options_, sift_gpu.get())) {
//            std::cerr << "ERROR: SiftGPU not fully supported." << std::endl;
//            return;
//        }
//    }
//
//    //extract features for the query images
//    for (int i = 0; i < queryImgs.size();i++) {
//        bool success = ExtractSiftFeaturesGPU(
//            sift_options_, queryImgs[i], sift_gpu.get(),
//            &kpts[i], &descripts[i]);
//    }
//}

