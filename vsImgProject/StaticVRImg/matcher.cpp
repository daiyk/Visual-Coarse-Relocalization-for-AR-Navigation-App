#include "matcher.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <filesystem>
#include "extractor.h"
#include "fileManager.h"

extern "C" {
    #include "vl/kdtree.h"
}
using params = fileManager::parameters;
using namespace cv;
namespace fs = std::filesystem;
std::vector<DMatch> matcher::kdTree(Mat& source, Mat& query) {
    if (!source.isContinuous() || !query.isContinuous())
    {
        throw std::runtime_error("ERROR: source descriptors or query descriptor are not continuous, matching function terminated");
    }
    std::vector<DMatch> matches;
    std::cout << "start building kd-tree for matching......" << std::endl;
    //kd-tree building
    int dim = 128;
    int numOfTree = 1;
    VlKDForest* tree = vl_kdforest_new(VL_TYPE_FLOAT, dim, numOfTree, VlDistanceL2);

    //start building tree build from the first image
    int numWords = source.rows;
    int numQuery = query.rows;

    //use img1 to build kd-tree
    vl_kdforest_build(tree, numWords, source.ptr<float>(0));

    //set searcher
    //use std vector instead

    std::vector<vl_uint32> NNs(params::numOfNN * numQuery);
    std::vector<float> NNdist(params::numOfNN * numQuery);

    //vl_uint32* NNs = (vl_uint32*)vl_malloc(params::numOfNN * sizeof(vl_uint32) * numQuery);
    //float* NNdist = (float*)vl_malloc(params::numOfNN * sizeof(float) * numQuery);

    VlKDForestSearcher* searcher = vl_kdforest_new_searcher(tree);
    vl_kdforest_set_thresholding_method(tree, VL_KDTREE_MEDIAN); //use median as the criteria
    vl_kdforest_set_max_num_comparisons(tree, params::maxNumComp); // set max num of comparison
    int numOfleaf = vl_kdforest_query_with_array(tree, NNs.data(), params::numOfNN, numQuery, NNdist.data(), query.ptr<float>(0));
    std::cout << " -> total number of " << numOfleaf << " leafs are visited during kd-tree building" << std::endl;

    //build Dmatches and check the distance ratio to avoid false matching
    for (int i = 0; i < numQuery; i++)
    {
        //compare the first and second nn distance
        double one2TwoNN = NNdist[params::numOfNN * i] / NNdist[params::numOfNN * i + 1];
        if (one2TwoNN <= params::distRat) { //use distance ratio to filter the matching keypoints
            DMatch match = DMatch(i, NNs[params::numOfNN * i], NNdist[params::numOfNN * i]);
            matches.push_back(match);
        }
    }
    return matches;
}
matcher::matchOut matcher::kdTreeDemo(std::string& img1, std::string& img2, bool display) {
    std::vector<std::vector<KeyPoint>> ttl_kpts;
    int dim = 128;
    Mat outImg;
    std::vector<std::string> trainPath; //template path store the image paths
    std::vector<DMatch> matches; //
    matcher::matchOut matchRel;
    
    trainPath.push_back(img1);
    Mat descripts1,descripts2;
    std::vector<KeyPoint> kpts1,kpts2;
    extractor::vlimg_descips_compute(trainPath, descripts1, kpts1);

    trainPath.pop_back();
    trainPath.push_back(img2);
    extractor::vlimg_descips_compute(trainPath, descripts2, kpts2);

    //kd-tree building

    //VlKDForest* tree = vl_kdforest_new(VL_TYPE_FLOAT, dim, 1, VlDistanceL2);
    matches=matcher::kdTree(descripts1, descripts2);
    ////start building tree build from the first image
    //int numKpts1 = kpts1.size();
    //int numKpts2 = kpts2.size();

    //if (descripts1.isContinuous() && descripts2.isContinuous()) {
    //    //use img1 to build kd-tree
    //    vl_kdforest_build(tree, numKpts1, descripts1.ptr<float>(0));

    //    //set searcher
    //    vl_uint32* NNs = (vl_uint32*)vl_malloc(params::numOfNN * sizeof(vl_uint32) * numKpts2);
    //    float* NNdist = (float*)vl_malloc(params::numOfNN * sizeof(float) * numKpts2);

    //    VlKDForestSearcher* searcher = vl_kdforest_new_searcher(tree);
    //    vl_kdforest_set_thresholding_method(tree, VL_KDTREE_MEDIAN); //use median as the criteria

    //    int numOfleaf = vl_kdforest_query_with_array(tree, NNs, 2, numKpts2, NNdist, descripts2.ptr<float>(0));

    //    std::cout << " -> total number of " << numOfleaf << " leafs are visited" << std::endl;
    //    //draw matches
    //    

    //    //build Dmatches
    //    for (int i = 0; i < numKpts2; i++)
    //    {
    //        DMatch match = DMatch(i, NNs[params::numOfNN * i], NNdist[params::numOfNN * i]);
    //        matches.push_back(match);
    //    }
    Mat image1 = imread(img1), image2 = imread(img2);
    matchRel.matches = matches;
    matchRel.source = kpts2;
    matchRel.refer = kpts1;
    if (display) {
        drawMatches(image2, kpts2, image1, kpts1, matches, outImg, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        namedWindow("Matches", WINDOW_NORMAL);
        imshow("Matches", outImg);
        waitKey();
    }
   /* else {
        std::cout << "ERROR: Matching stop for descriptors Mat is not continuous" << std::endl;
    }*/
    return matchRel;
}


