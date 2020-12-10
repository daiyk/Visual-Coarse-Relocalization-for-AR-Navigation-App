#include "vlad.h"
#include "fileManager.h"
#include "extractor.h"
#include "cluster.h"
#include "matcher.h"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using params = fileManager::parameters;
using namespace cv;
bool useCovdet = false;
vlad::vlad::vlad(std::vector<std::string>& paths) :tree(nullptr) {
    //read imgs from files and extracts features

    Mat allDescripts,centers;
    std::vector<int> n_descrips;
    int numQuery = paths.size();
    n_descrips.reserve(numQuery);
    std::vector<cv::KeyPoint> cv_keypoints;
    this->enc.release();
    
    //iterate through all paths and record descriptors
    for (int i = 0; i < numQuery; i++) {
        if (!fs::exists(fs::path(paths[i]))) {
            std::cout << "vlimg_descrip_compute: Warning: " << paths[i] << " does not exist!" << std::endl;
            continue;
        }
        cv::Mat descripts1;
        std::vector<cv::KeyPoint> kpts1;
        cv::Mat grayImg;
        
        cv::cvtColor(cv::imread(paths[i]), grayImg, cv::COLOR_BGR2GRAY);
        if (useCovdet)
            extractor::covdetSIFT(grayImg, descripts1, kpts1);
        else
        {
            extractor::vlimg_descips_compute_simple(grayImg, descripts1, kpts1);
        }
        
        n_descrips.push_back(descripts1.rows);
        allDescripts.push_back(descripts1);
        std::cout << i << " th image is finsihed extraction\n";
    }

    //all kmeans parameters setting corresponding with OpenCV setting
    int vladCenters = params::vladCenters;
    cluster::vl_visual_word_compute(allDescripts, centers, vladCenters);

    //build kdtree
    if (!centers.isContinuous() || !allDescripts.isContinuous())
    {
        throw std::runtime_error("ERROR: source descriptors or query descriptor are not continuous, matching function terminated");
    }
    this->kCenters = centers;

    //write vlad center to file
    std::vector<cv::KeyPoint> kpts;
    fileManager::write_to_file("vlad_" + std::to_string(params::vladCenters), kpts, centers);

    //kd-tree building
    int dim = params::descriptDim;
    int numOfTree = 1;
    this->tree = vl_kdforest_new(VL_TYPE_FLOAT, dim, numOfTree, VlDistanceL2);

    vl_kdforest_build(tree, vladCenters, kCenters.ptr<float>(0));
    vl_kdforest_set_thresholding_method(tree, VL_KDTREE_MEDIAN); //use median as the criteria
    vl_kdforest_set_max_num_comparisons(tree, params::maxNumComp); // set max num of comparison
    //iterate through every img's descriptors and record the encoding vectors
    int cur_index = 0;
    for (int i = 0; i < numQuery; i++) {
        int num_query_descrips = n_descrips[i];
        std::vector<vl_uint32> NNs(num_query_descrips);
        std::vector<float> NNdist(num_query_descrips);

        //vl_uint32* NNs = (vl_uint32*)vl_malloc(params::numOfNN * sizeof(vl_uint32) * numQuery);
        //float* NNdist = (float*)vl_malloc(params::numOfNN * sizeof(float) * numQuery);
        int numOfleaf = vl_kdforest_query_with_array(tree, NNs.data(), 1, num_query_descrips, NNdist.data(), allDescripts.ptr<float>(cur_index));

        //build assignment and encode the imageset in vald vectors
        std::vector<float> assignments(num_query_descrips * vladCenters, 0);
        for (int i = 0; i < num_query_descrips; i++) {
            assignments[i * vladCenters + NNs[i]] = 1.;
        }
        cv::Mat1f enc(vladCenters, params::descriptDim, float(0.0));

        /*std::vector<float> enc(vladCenters * params::descriptDim);*/

        //encoding with vlad and store the encoding vector
        vl_vlad_encode(enc.ptr<float>(0), VL_TYPE_FLOAT, kCenters.ptr<float>(0), params::descriptDim, vladCenters, allDescripts.ptr<float>(cur_index), num_query_descrips, assignments.data(), VL_VLAD_FLAG_NORMALIZE_COMPONENTS);
        cur_index += num_query_descrips;
       
        //encode encoding to the Mat
        this->enc.push_back(enc);
    }
    this->write_to_file("vlad_" + std::to_string(params::vladCenters));
}

//read the pretrained centers and encoding vector
vlad::vlad::vlad(std::string centerPath, std::string encPath):tree(nullptr) {
    //read kcenters and encoding vector
    //read kcenters
    cv::FileStorage reader;
    reader.open(centerPath, cv::FileStorage::READ);
    if (!reader.isOpened()) { std::cout << "ERROR: failed to open the vlad kcenter file" << std::endl; return; }

    //read kcenters
    reader["kcenters"] >> this->kCenters;
    reader.release();

    reader.open(encPath, cv::FileStorage::READ);
    if (!reader.isOpened()) { std::cout << "ERROR: failed to open the vladEnc file" << std::endl; return; }
    reader["vladEnc"] >> this->enc;

    //build tree
    int dim = params::descriptDim;
    int numOfTree = 1;
    this->tree = vl_kdforest_new(VL_TYPE_FLOAT, dim, numOfTree, VlDistanceL2);

    vl_kdforest_build(tree, params::vladCenters, kCenters.ptr<float>(0));
    vl_kdforest_set_thresholding_method(tree, VL_KDTREE_MEDIAN); //use median as the criteria
    vl_kdforest_set_max_num_comparisons(tree, params::maxNumComp); // set max num of comparison
}
vlad::vlad::vlad(std::vector<cv::Mat> descripts): tree(nullptr) {
    int vladCenters = params::vladCenters;
    cv::Mat centers;//centers mat
    
    //rebuild descripts to a single mat
    cv::Mat allDescripts;
    for (int i = 0; i < descripts.size(); i++) {
        allDescripts.push_back(descripts[i]);
    }
    cluster::vl_visual_word_compute(allDescripts, centers, vladCenters);
    
    if (!centers.isContinuous() || !allDescripts.isContinuous())
    {
        throw std::runtime_error("ERROR: source descriptors or query descriptor are not continuous, matching function terminated");
    }
    allDescripts.release();
    this->kCenters = centers;
    //write vlad center to file
    std::vector<cv::KeyPoint> kpts;
    fileManager::write_to_file("vlad_" + std::to_string(params::vladCenters), kpts, centers);

    //kd-tree building
    int dim = params::descriptDim;
    int numOfTree = 1;
    this->tree = vl_kdforest_new(VL_TYPE_FLOAT, dim, numOfTree, VlDistanceL2);

    vl_kdforest_build(tree, vladCenters, kCenters.ptr<float>(0));
    vl_kdforest_set_thresholding_method(tree, VL_KDTREE_MEDIAN); //use median as the criteria
    vl_kdforest_set_max_num_comparisons(tree, params::maxNumComp); // set max num of comparison
    
    for (int i = 0; i < descripts.size(); i++) {
        int num_query_descrips = descripts[i].rows;
        std::vector<vl_uint32> NNs(num_query_descrips);
        std::vector<float> NNdist(num_query_descrips);

        //vl_uint32* NNs = (vl_uint32*)vl_malloc(params::numOfNN * sizeof(vl_uint32) * numQuery);
        //float* NNdist = (float*)vl_malloc(params::numOfNN * sizeof(float) * numQuery);
        int numOfleaf = vl_kdforest_query_with_array(tree, NNs.data(), 1, num_query_descrips, NNdist.data(), descripts[i].ptr<float>(0));

        //build assignment and encode the imageset in vlad vectors
        std::vector<float> assignments(num_query_descrips * vladCenters, 0);
        for (int i = 0; i < num_query_descrips; i++) {
            assignments[i * vladCenters + NNs[i]] = 1.;
        }
        cv::Mat1f enc(vladCenters, params::descriptDim, float(0.0));

        /*std::vector<float> enc(vladCenters * params::descriptDim);*/

        //encoding with vlad and store the encoding vector
        vl_vlad_encode(enc.ptr<float>(0), VL_TYPE_FLOAT, kCenters.ptr<float>(0), params::descriptDim, vladCenters, descripts[i].ptr<float>(0), num_query_descrips, assignments.data(), VL_VLAD_FLAG_NORMALIZE_COMPONENTS);

        //encode encoding to the Mat
        this->enc.push_back(enc);

        std::cout << i << "th image finished vlad encoding\n";
    }
    this->write_to_file("vlad_" + std::to_string(params::vladCenters));
}
vlad::vlad::~vlad() {
    vl_kdforest_delete(this->tree);
    this->enc.release();
    this->kCenters.release();
}

void vlad::vlad::searchWithDescripts(cv::Mat query_descript, std::vector<int>& ind, std::vector<double>& score, int bestOfAll) {
    if (!this->tree) {
        throw std::invalid_argument("VLAD: Tree is not set search failed!");
    }

    int num_descrips = query_descript.rows;
    std::vector<vl_uint32> NNs(num_descrips);
    std::vector<float> NNdist(num_descrips);

    //vl_uint32* NNs = (vl_uint32*)vl_malloc(params::numOfNN * sizeof(vl_uint32) * numQuery);
    //float* NNdist = (float*)vl_malloc(params::numOfNN * sizeof(float) * numQuery);
    int numOfleaf = vl_kdforest_query_with_array(this->tree, NNs.data(), 1, num_descrips, NNdist.data(), query_descript.ptr<float>(0));

    //build assignment and encode the imageset in vlad vectors
    std::vector<float> assignments(num_descrips * params::vladCenters, 0);
    for (int i = 0; i < num_descrips; i++) {
        assignments[i * params::vladCenters + NNs[i]] = 1.;
    }
    cv::Mat1f query_enc(params::vladCenters, params::descriptDim, float(0.0));

    //encoding with vlad and store the encoding vector
    vl_vlad_encode(query_enc.ptr<float>(0), VL_TYPE_FLOAT, kCenters.ptr<float>(0), params::descriptDim, params::vladCenters, query_descript.ptr<float>(0), num_descrips, assignments.data(), VL_VLAD_FLAG_NORMALIZE_COMPONENTS);
    //search the index of img that is minimal to the 
    this->enc_index(query_enc, ind, score, bestOfAll);

}


void vlad::vlad::search(cv::Mat img, std::vector<int>& ind, std::vector<double>& score, int bestOfAll) {
    //extract feature by vlfeat

    if (!this->tree) {
        throw std::invalid_argument("VLAD: Tree is not set search failed!");
    }

    cv::Mat descripts1;
    std::vector<cv::KeyPoint> kpts1;
    try{ 
        if (useCovdet)
            extractor::covdetSIFT(img, descripts1, kpts1);
        else
        {
            extractor::vlimg_descips_compute_simple(img, descripts1, kpts1);
        }
        
    }
    catch (std::invalid_argument& e) {
        std::cout << e.what() << std::endl;
        return;
    };
    int num_descrips = descripts1.rows;
    std::vector<vl_uint32> NNs(num_descrips);
    std::vector<float> NNdist(num_descrips);

    //vl_uint32* NNs = (vl_uint32*)vl_malloc(params::numOfNN * sizeof(vl_uint32) * numQuery);
    //float* NNdist = (float*)vl_malloc(params::numOfNN * sizeof(float) * numQuery);
    int numOfleaf = vl_kdforest_query_with_array(this->tree, NNs.data(), 1, num_descrips, NNdist.data(), descripts1.ptr<float>(0));

    //build assignment and encode the imageset in vald vectors
    std::vector<float> assignments(num_descrips * params::vladCenters, 0);
    for (int i = 0; i < num_descrips; i++) {
        assignments[i * params::vladCenters + NNs[i]] = 1.;
    }
    cv::Mat1f query_enc(params::vladCenters, params::descriptDim, float(0.0));

    //encoding with vlad and store the encoding vector
    vl_vlad_encode(query_enc.ptr<float>(0), VL_TYPE_FLOAT, kCenters.ptr<float>(0), params::descriptDim, params::vladCenters, descripts1.ptr<float>(0), num_descrips, assignments.data(), VL_VLAD_FLAG_NORMALIZE_COMPONENTS);
    //search the index of img that is minimal to the 
    this->enc_index(query_enc,ind,score,bestOfAll);

    //read from the path for test image
  /*  Mat allDescripts, kCenters;
    std::vector<cv::KeyPoint> cv_keypoints;
    extractor::vlimg_descips_compute(paths, allDescripts, cv_keypoints);*/

}

void vlad::vlad::enc_index(Mat1f& query, std::vector<int> &ind, std::vector<double> &score, int bestOfAll) {
    if (this->enc.empty()) {
        throw std::invalid_argument("VLAD: access encoding vectors before initialize it");
    }
    //iterate through the enc and find the corresponding imgs
    ind.clear();
    ind.reserve(bestOfAll);
    score.clear();
    score.reserve(bestOfAll);
    std::vector<double> scores;
   
    int rows = this->enc.rows;
    scores.reserve(rows / params::vladCenters);
    for (int i = 0;i * params::vladCenters < rows; i++) {
        //
        cv::Mat temp(this->enc, Range(i * params::vladCenters, (i + 1) * params::vladCenters));
        
        //iterate through and compute the best fits
        double i_score = cv::norm(temp, query, NORM_L2);
        scores.push_back(i_score);
    }

    //find the best scores
    std::vector<int> score_index(scores.size());
    //cout score function
    /*std::cout << "before index rerank: "<<std::endl;
    for (int j = 0; j < scores[i].size(); j++) {
        std::cout << scores[i][j] << " ";
    }*/
    //scores represents the distance, thus need to rank it from small to large
    std::iota(score_index.begin(), score_index.end(), 0);
    std::sort(score_index.begin(), score_index.end(), [&](size_t left, size_t right) {return scores[left] < scores[right]; });

    //report temp after rerank
    /*for (int j = 0; j < temp.size(); j++) {
        std::cout << temp[j] << " ";
    }*/

    //return score and index
    for (int i = 0; i < bestOfAll; i++) {
        ind.push_back(score_index[i]);
        score.push_back(scores[score_index[i]]);
    }
}

void vlad::vlad::write_to_file(std::string name) {
    if (!this->enc.empty()) {
        cv::FileStorage filewriter("Result/" + name + "_vladEnc.yml", cv::FileStorage::WRITE);
        filewriter << "vladEnc" << this->enc;
        std::cout << "  --> vlad encoding vector this->enc are written to file " << name << "_vladEnc.yml......" << std::endl;
    }
    else {
        std::cout << "vlad::write_to_file: object's encoding vector is empty! abort writing" << std::endl;
    }
}