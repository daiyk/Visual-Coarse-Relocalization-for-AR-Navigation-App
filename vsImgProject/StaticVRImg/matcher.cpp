#include "matcher.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "constAndTypes.h"
#include "extractor.h"
#include <filesystem>
extern "C" {
    #include "vl/kdtree.h"
}

namespace cAt = constandtypes;
using namespace cv;
namespace fs = std::filesystem;
matcher::matchOut matcher::kdTree(std::string& img1, std::string& img2, bool display) {
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

    VlKDForest* tree = vl_kdforest_new(VL_TYPE_FLOAT, dim, 1, VlDistanceL2);

    //start building tree build from the first image
    int numKpts1 = kpts1.size();
    int numKpts2 = kpts2.size();

    if (descripts1.isContinuous() && descripts2.isContinuous()) {
        //use img1 to build kd-tree
        vl_kdforest_build(tree, numKpts1, descripts1.ptr<float>(0));

        //set searcher
        vl_uint32* NNs = (vl_uint32*)vl_malloc(cAt::numOfNN * sizeof(vl_uint32) * numKpts2);
        float* NNdist = (float*)vl_malloc(cAt::numOfNN * sizeof(float) * numKpts2);

        VlKDForestSearcher* searcher = vl_kdforest_new_searcher(tree);
        vl_kdforest_set_thresholding_method(tree, VL_KDTREE_MEDIAN); //use median as the criteria

        int numOfleaf = vl_kdforest_query_with_array(tree, NNs, 2, numKpts2, NNdist, descripts2.ptr<float>(0));

        std::cout << " -> total number of " << numOfleaf << " leafs are visited" << std::endl;
        //draw matches
        Mat image1 = imread(img1), image2 = imread(img2);

        //build Dmatches
        for (int i = 0; i < numKpts2; i++)
        {
            DMatch match = DMatch(i, NNs[cAt::numOfNN * i], NNdist[cAt::numOfNN * i]);
            matches.push_back(match);
        }
        matchRel.matches = matches;
        matchRel.source = kpts2;
        matchRel.refer = kpts1;
        if (display) {
            drawMatches(image2, kpts2, image1, kpts1, matches, outImg, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            namedWindow("Matches", WINDOW_NORMAL);
            imshow("Matches", outImg);
            waitKey();
        }
    }
    else {
        std::cout << "ERROR: Matching stop for descriptors Mat is not continuous" << std::endl;
    }
    return matchRel;
}


