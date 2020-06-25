#include "vlad.h"
#include "fileManager.h"
#include "extractor.h"
#include "cluster.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using params = fileManager::parameters;
using namespace cv;
vlad::vlad::vlad(std::vector<std::string>& paths, int numOfCenter) :tree(nullptr) {
    //read imgs from files and extracts features

    Mat allDescripts, centers;
    std::vector<cv::KeyPoint> cv_keypoints;
    extractor::vlimg_descips_compute(paths, allDescripts, cv_keypoints);

    std::vector<KeyPoint>().swap(cv_keypoints);
    //all kmeans parameters setting corresponding with OpenCV setting
    int tempCenter = params::centers;
    params::centers = numOfCenter;
    cluster::vl_visual_word_compute(allDescripts, centers);
    params::centers = tempCenter;

    //build kdtree

    if (!centers.isContinuous() || !allDescripts.isContinuous())
    {
        throw std::runtime_error("ERROR: source descriptors or query descriptor are not continuous, matching function terminated");
    }
    this->kCenters = centers;
    std::vector<DMatch> matches;

    //kd-tree building
    int dim = params::descriptDim;
    int numOfTree = 1;
    this->tree = vl_kdforest_new(VL_TYPE_FLOAT, dim, numOfTree, VlDistanceL2);

    int numWords = kCenters.rows;
    int numQuery = allDescripts.rows;

    vl_kdforest_build(tree, numWords, kCenters.ptr<float>(0));

    //set searcher
    //use std vector instead
    std::vector<vl_uint32> NNs(numQuery);
    std::vector<float> NNdist(numQuery);

    //vl_uint32* NNs = (vl_uint32*)vl_malloc(params::numOfNN * sizeof(vl_uint32) * numQuery);
    //float* NNdist = (float*)vl_malloc(params::numOfNN * sizeof(float) * numQuery);

    VlKDForestSearcher* searcher = vl_kdforest_new_searcher(tree);
    vl_kdforest_set_thresholding_method(tree, VL_KDTREE_MEDIAN); //use median as the criteria
    vl_kdforest_set_max_num_comparisons(tree, params::maxNumComp); // set max num of comparison
    int numOfleaf = vl_kdforest_query_with_array(tree, NNs.data(), 1, numQuery, NNdist.data(), allDescripts.ptr<float>(0));

    //build assignment and encode the imageset in vald vectors

    std::vector<float> assignments(numQuery * numOfCenter, 0);
    for (int i = 0; i < numQuery; i++) {
        assignments[i * numOfCenter + NNs[i]] = 1.;
    }
    std::vector<float> enc(numOfCenter * params::descriptDim);

    //encoding with vlad and store the encoding vector
    vl_vlad_encode(enc.data(), VL_TYPE_FLOAT, kCenters.ptr<float>(0), params::descriptDim, numOfCenter, allDescripts.ptr<float>(0), numQuery, assignments.data(), 0);
    this->enc = enc;
}


int vlad::vlad::search(cv::Mat img) {
    //std::vector<vl_uint32> NNs(numQuery);
    //std::vector<float> NNdist(numQuery);

    ////vl_uint32* NNs = (vl_uint32*)vl_malloc(params::numOfNN * sizeof(vl_uint32) * numQuery);
    ////float* NNdist = (float*)vl_malloc(params::numOfNN * sizeof(float) * numQuery);

    //VlKDForestSearcher* searcher = vl_kdforest_new_searcher(tree);
    //vl_kdforest_set_thresholding_method(tree, VL_KDTREE_MEDIAN); //use median as the criteria
    //vl_kdforest_set_max_num_comparisons(tree, params::maxNumComp); // set max num of comparison
    //int numOfleaf = vl_kdforest_query_with_array(tree, NNs.data(), 1, numQuery, NNdist.data(), allDescripts.ptr<float>(0));

    //read from the path for test image
  /*  Mat allDescripts, kCenters;
    std::vector<cv::KeyPoint> cv_keypoints;
    extractor::vlimg_descips_compute(paths, allDescripts, cv_keypoints);*/

}