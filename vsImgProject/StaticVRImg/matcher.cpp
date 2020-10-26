#include "matcher.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include "extractor.h"
#include "fileManager.h"
#include "cluster.h"
#include "helper.h"

extern "C" {
    #include "vl/kdtree.h"
    #include "vl/kmeans.h"
}
using params = fileManager::parameters;
namespace fs = boost::filesystem;

matcher::kdTree::kdTree(cv::Mat source) {
    if (!source.isContinuous())
    {
        throw std::runtime_error("ERROR: source descriptors are not continuous, kdTree initializer terminated");
    }
    if (source.rows == 0) {
        throw std::runtime_error("Error: source mat is empty");
    }

    source.copyTo(this->vocab);
    //kd-tree building
    int dim = params::descriptDim;
    int numOfTree = 4;
    this->numOfNN = params::numOfNN;
    this->tree = vl_kdforest_new(VL_TYPE_FLOAT, dim, numOfTree, VlDistanceL2);
    
    //start building tree build from the first image
    this->vocab_size_ = source.rows;
    
    //check the types of source mat
    auto types = helper::cvtype2str(source.type());
    
    if (types.substr(0, 1) != "3") {
        //if it is uchar than transform it to float
        this->vocab.convertTo(this->vocab, CV_32F);
    }
    //build kd-tree
    vl_kdforest_build(this->tree, vocab_size_, this->vocab.ptr<float>(0));

    //set searcher
    vl_kdforest_set_thresholding_method(tree, VL_KDTREE_MEDIAN); //use median as the criteria
    vl_kdforest_set_max_num_comparisons(tree, params::maxNumComp); // set max num of comparison
}

matcher::kdTree::~kdTree() {
    vl_kdforest_delete(this->tree);
}

std::vector<cv::DMatch> matcher::kdTree::search(cv::Mat& query) {
    std::vector<cv::DMatch> matches;
    if (query.rows == 0 || !tree) {
        return matches;
    }
    if (!query.isContinuous())
    {
        throw std::runtime_error("ERROR: source descriptors or query descriptor are not continuous, matching function terminated");
    }
    //number of query descripts
    int numQuery = query.rows;

    //set search result container
    std::vector<vl_uint32> NNs(numOfNN * numQuery);
    std::vector<float> NNdist(numOfNN * numQuery);

    /*vl_uint32* NNs = (vl_uint32*)vl_malloc(params::numOfNN * sizeof(vl_uint32) * numQuery);
    float* NNdist = (float*)vl_malloc(params::numOfNN * sizeof(float) * numQuery);*/

    int numOfleaf = vl_kdforest_query_with_array(this->tree, NNs.data(), numOfNN, numQuery, NNdist.data(), query.ptr<float>(0));

    //build Dmatches and check the distance ratio to avoid false matching
    for (int i = 0; i < numQuery; i++)
    {
        //compare the first and second nn distance
        double one2TwoNN = NNdist[numOfNN * i] / NNdist[numOfNN * i + 1];
        if (one2TwoNN <= params::MATCH_THRES) { //use distance ratio to filter the matching keypoints
            cv::DMatch match = cv::DMatch(i, NNs[numOfNN * i], NNdist[numOfNN * i]);
            matches.push_back(match);
        }
    }
    return matches;
}
std::vector<cv::DMatch> matcher::kdTree::colmapSearch(cv::Mat& query) {
    std::vector<cv::DMatch> matches;
    cv::Mat query_transform;
    query.convertTo(query_transform, CV_32F);

    if (query.rows == 0) {
        return matches;
    }
    if (!query.isContinuous())
    {
        throw std::runtime_error("ERROR: source descriptors or query descriptor are not continuous, matching function terminated");
    }
    //number of query descripts
    int numQuery = query.rows;

    //set search result container
    std::vector<vl_uint32> NNs(this->numOfNN * numQuery);
    std::vector<float> NNdist(this->numOfNN * numQuery);

    int numOfleaf = vl_kdforest_query_with_array(tree, NNs.data(), this->numOfNN, numQuery, NNdist.data(), query_transform.ptr<float>(0));

    //build Dmatches and check the distance ratio to avoid false matching
    cv::Mat nearestDist(numQuery, 2, CV_32F, cv::Scalar(0));
    cv::Mat nearestInd(numQuery, 2, CV_32S, cv::Scalar(-1));
    const float kDistNorm = 1.f / (512.f * 512.f);
    for (int i = 0; i < numQuery; i++)
    {
        for (int j = 0; j < this->numOfNN; j++) {
            //compute and stores the distance
            float distance = query_transform.row(i).dot(this->vocab.row(NNs[this->numOfNN * i + j]));
            if (distance > nearestDist.at<float>(i, 0)) {
                nearestDist.at<float>(i, 0) = distance;
                nearestInd.at<int>(i, 0) = NNs[this->numOfNN * i + j];
                continue;
            }
            if (distance > nearestDist.at<float>(i, 1)) {
                nearestDist.at<float>(i, 1) = distance;
                nearestInd.at<int>(i, 1) = NNs[this->numOfNN * i + j];
            }
        }
        if (nearestInd.at<int>(i, 0) == -1) {
            continue;
        }
        nearestDist.at<float>(i, 0) = std::acos(std::min(kDistNorm * nearestDist.at<float>(i, 0), 1.f));
        nearestDist.at<float>(i, 1) = std::acos(std::min(kDistNorm * nearestDist.at<float>(i, 1), 1.f));

        //do ratio test because it is acos thus need to choose the smaller one
        if (nearestDist.at<float>(i, 0) >= params::MATCH_THRES * nearestDist.at<float>(i, 1)) {
            continue;
        }
        cv::DMatch match = cv::DMatch(i, nearestInd.at<int>(i, 0), nearestDist.at<float>(i, 0));
        matches.push_back(match);
    }
    return matches;
}

matcher::matchOut matcher::kdTree::kdTreeDemo(std::string& img1, std::string& img2, bool display) {
    std::vector<std::vector<cv::KeyPoint>> ttl_kpts;
    int dim = params::descriptDim;
    cv::Mat outImg;
    std::vector<std::string> trainPath; //template path store the image paths
    std::vector<cv::DMatch> matches; //
    matcher::matchOut matchRel;
    
    trainPath.push_back(img1);
    cv::Mat descripts1,descripts2;
    std::vector<cv::KeyPoint> kpts1,kpts2;
    extractor::vlimg_descips_compute(trainPath, descripts1, kpts1);

    trainPath.pop_back();
    trainPath.push_back(img2);
    extractor::vlimg_descips_compute(trainPath, descripts2, kpts2);

    //kd-tree building
    
    //descripts1 as source
    
    matcher::kdTree test(descripts1);
    matches = test.search(descripts2);
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
    cv::Mat image1 = cv::imread(img1), image2 = cv::imread(img2);
    matchRel.matches = matches;
    matchRel.source = kpts2;
    matchRel.refer = kpts1;
    if (display) {
        cv::drawMatches(image2, kpts2, image1, kpts1, matches, outImg, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow("Matches", cv::WINDOW_NORMAL);
        imshow("Matches", outImg);
        cv::waitKey();
    }

    return matchRel;
}

std::vector<cv::DMatch> matcher::opencvFlannMatcher(cv::Mat& source, cv::Mat& query) {
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch(query, source, knn_matches, 2);

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = params::MATCH_THRES;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    return good_matches;
}

void matcher::RANSC(cv::Mat& sourceDescrips, std::vector<cv::KeyPoint> &sourcekpts, cv::Mat& queryDescrips, std::vector<cv::KeyPoint>& querykpts, cv::Mat& mask, cv::Mat& homo) {
    matcher::kdTree matches(sourceDescrips);
    auto bestMatches = matches.search(queryDescrips);

    //records the matching keypoints by the matching result
    std::vector<cv::Point2f> matchkpts1;
    std::vector<cv::Point2f> matchkpts2;
    for (int i = 0; i < bestMatches.size(); i++) {
        matchkpts1.push_back(sourcekpts[bestMatches[i].trainIdx].pt);
        matchkpts2.push_back(querykpts[bestMatches[i].queryIdx].pt);
    }
    homo = cv::findHomography(matchkpts1, matchkpts2, mask, cv::RANSAC);
}

std::vector<cv::DMatch> matcher::colmapFlannMatcher(const cv::Mat& query_descriptors, const cv::Mat& database_descriptors, int NNeighbors) {
    cv::Mat indices_1to2, distance;
    distance.resize(query_descriptors.rows);
    std::vector<cv::DMatch> matches;
    if (query_descriptors.rows == 0 || database_descriptors.rows == 0) {
        return matches;
    }

    if (!database_descriptors.isContinuous())
    {
        throw std::runtime_error("ERROR: source descriptors are not continuous, kdTree initializer terminated");
    }
    //kd-tree building
    int dim = params::descriptDim;
    int numOfTree = 4;
    auto tree = vl_kdforest_new(VL_TYPE_FLOAT, dim, numOfTree, VlDistanceL2);
    //start building tree build from the database 
    int numWords = database_descriptors.rows;
    cv::Mat db_transform, query_transform;
    database_descriptors.convertTo(db_transform, CV_32F);
    query_descriptors.convertTo(query_transform, CV_32F);
    
    //build kd-tree
    vl_kdforest_build(tree, numWords, db_transform.ptr<float>(0));

    //set searcher
    vl_kdforest_set_thresholding_method(tree, VL_KDTREE_MEDIAN); //use median as the criteria
    vl_kdforest_set_max_num_comparisons(tree, params::maxNumComp); // set max num of comparison

    
    if (query_descriptors.rows == 0) {
        return matches;
    }
    if (!query_descriptors.isContinuous())
    {
        throw std::runtime_error("ERROR: source descriptors or query descriptor are not continuous, matching function terminated");
    }
    //number of query descripts
    int numQuery = query_descriptors.rows;

    //set search result container
    std::vector<vl_uint32> NNs(NNeighbors * numQuery);
    std::vector<float> NNdist(NNeighbors * numQuery);

    int numOfleaf = vl_kdforest_query_with_array(tree, NNs.data(), NNeighbors, numQuery, NNdist.data(), query_transform.ptr<float>(0));

    //build Dmatches and check the distance ratio to avoid false matching
    cv::Mat nearestDist(numQuery, 2, CV_32F, cv::Scalar(0));
    cv::Mat nearestInd(numQuery, 2, CV_32S, cv::Scalar(-1));
    const float kDistNorm = 1.f / (512.f * 512.f);
    for (int i = 0; i < numQuery; i++)
    {
        for (int j = 0; j < NNeighbors; j++) {
            //compute and stores the distance
            float distance = query_transform.row(i).dot(db_transform.row(NNs[NNeighbors * i + j]));
            if (distance > nearestDist.at<float>(i, 0)) {
                nearestDist.at<float>(i, 0) = distance;
                nearestInd.at<int>(i, 0) = NNs[NNeighbors * i + j];
                continue;
            }
            if (distance > nearestDist.at<float>(i, 1)) {
                nearestDist.at<float>(i, 1) = distance;
                nearestInd.at<int>(i, 1) = NNs[NNeighbors * i + j];
            }
        }
        if (nearestInd.at<int>(i, 0) == -1) {
            continue;
        }
        nearestDist.at<float>(i, 0) = std::acos(std::min(kDistNorm * nearestDist.at<float>(i, 0), 1.f));
        nearestDist.at<float>(i, 1) = std::acos(std::min(kDistNorm * nearestDist.at<float>(i, 1), 1.f));

        //do ratio test because it is acos thus need to choose the smaller one
        if (nearestDist.at<float>(i, 0) >= params::MATCH_THRES * nearestDist.at<float>(i, 1)) {
            continue;
        }
        cv::DMatch match = cv::DMatch(i, nearestInd.at<int>(i, 0), nearestDist.at<float>(i, 0));
        matches.push_back(match);
    }
    return matches;
}



//explicit template initialization

