#include "helper.h"

cv::Mat helper::bitmapToMat(colmap::Bitmap& bitmap) {
    cv::Mat imgBitmap(bitmap.Height(), bitmap.Width(), CV_8U);
    auto dataFloat = bitmap.ConvertToRowMajorArray();

    //transform bitmap to opencv image
    for (int i = 0; i < bitmap.Height(); i++) {
        for (int j = 0; j < bitmap.Width(); j++) {
            imgBitmap.at<uchar>(i, j) = dataFloat[j + i * bitmap.Width()];
        }
    }
    return imgBitmap;
}
cv::Mat helper::DescriptorFloatToUint(cv::Mat descripts) {
    cv::Mat unsignedDescripts(descripts.rows, descripts.cols, CV_8U);
    for (int i = 0;i< descripts.rows; i++) {
        for(int j=0;j<descripts.cols;j++){
            const float value = std::round(512.0f * descripts.at<float>(i, j));
            unsignedDescripts.at<uint8_t>(i,j) = std::min(static_cast<float>(std::numeric_limits<uint8_t>::max()),std::max(static_cast<float>(std::numeric_limits<uint8_t>::min()), value));
            //cast value to half?
        }
    }
    return unsignedDescripts;
}

std::string helper::dateTime() {
    std::time_t t = std::time(0);   // get time now
    std::tm* now = std::localtime(&t);
    return std::to_string(now->tm_hour) + std::to_string(now->tm_min) + std::to_string(now->tm_sec);
}

//unpack octave number, copy from opencv source code https://github.com/opencv/opencv_contrib/blob/bebfd717485c79644a49ac406b0d5f717b881aeb/modules/xfeatures2d/src/sift.cpp#L214-L220
void helper::unpackOctave(const cv::KeyPoint& kpt, int& octave, int& layer, float& scale)
{
    octave = kpt.octave & 255;
    layer = (kpt.octave >> 8) & 255;
    octave = octave < 128 ? octave : (-128 | octave);
    scale = octave >= 0 ? 1.f / (1 << octave) : (float)(1 << -octave);
}

bool helper::isEqual(double x, double y)
{
    double maxXYOne = std::max({ 1.0, std::fabs(x) , std::fabs(y) });
    return std::fabs(x - y) <= std::numeric_limits<double>::epsilon() * maxXYOne;
}
void helper::computeScore1(std::vector<std::vector<double>>& raw_scores, std::vector<size_t>& edge_nums, std::vector<double>& raw_self_scores,bool tfidfWeight) {
    int n_query = raw_scores.size();
    int n_database = raw_scores[0].size();
    for (int i = 0; i < n_query; i++)
        for (int j = 0; j < n_database; j++) {
            if (edge_nums[i] + edge_nums[j + n_query] != 0) {
                if (!isEqual(raw_self_scores[i], 0.0) && !isEqual(raw_self_scores[j + n_query],0.0)) {
                    double edge_norm = 1.0 / (edge_nums[i] + edge_nums[j + n_query]);
                    double selfNormi, selfNormj;
                    if (tfidfWeight) {
                        selfNormi = raw_self_scores[i];
                        selfNormj = raw_self_scores[j + n_query];
                    }
                    else
                    {
                        selfNormi = raw_self_scores[i] * edge_norm * edge_norm;
                        selfNormj = raw_self_scores[j + n_query] * edge_norm * edge_norm;
                    }
                        
                        raw_scores[i][j] = raw_scores[i][j] / std::sqrt(selfNormi * selfNormj);
                }
            }
        }

}

void helper::computeScore2(std::vector<std::vector<double>>& raw_scores, std::vector<size_t>& edge_nums, std::vector<double>& raw_self_scores) {
    int n_query = raw_scores.size();
    int n_database = raw_scores[0].size();
    for (int i = 0; i < n_query; i++)
        for (int j = 0; j < n_database; j++) {
            if (edge_nums[i] + edge_nums[j + n_query] != 0) {
                double edge_norm = 0.0;
                edge_norm = 1.0 / (edge_nums[i] + edge_nums[j + n_query]);
                raw_scores[i][j] = raw_scores[i][j] / (raw_self_scores[i] * edge_norm * edge_norm);
            }
        }
}
void helper::computeScore3(std::vector<std::vector<double>>& raw_scores, std::vector<double>& raw_self_scores) {
    int n_query = raw_scores.size();
    int n_database = raw_scores[0].size();
    for (int i = 0; i < n_query; i++)
        for (int j = 0; j < n_database; j++) {
            if (!isEqual(raw_self_scores[i], 0.0) && !isEqual(raw_self_scores[j + n_query], 0.0)) {
                double selfNormi, selfNormj;          
                selfNormi = raw_self_scores[i];
                selfNormj = raw_self_scores[j + n_query];
                raw_scores[i][j] = raw_scores[i][j] / std::sqrt(selfNormi * selfNormj);
            }           
        }
}

std::vector<cv::KeyPoint> helper::colmapToCvKpts(colmap::FeatureKeypoints& kpts) {
    std::vector<cv::KeyPoint> cv_kpts;
    cv_kpts.resize(kpts.size());
    for (int i = 0; i < kpts.size(); i++) {
        cv_kpts.push_back(cv::KeyPoint(kpts[i].x, kpts[i].y,kpts[i].ComputeScale()));
    }
    return cv_kpts;
}

void helper::ExtractTopFeatures(colmap::FeatureKeypoints* keypoints, colmap::FeatureVisualIDs* ids, const size_t num_features=-1) {

    if (keypoints->size() != ids->ids.rows()) {
        throw std::invalid_argument("Error: visual ids and keypoints are not in the same length.");
    }
    if (num_features == 0) {
        throw std::invalid_argument("Error: num of features need to be non-zero");
    }

    if (static_cast<size_t>(ids->ids.rows()) <= num_features || num_features==-1) {
        return;
    }

    colmap::FeatureKeypoints top_scale_keypoints;
    colmap::FeatureVisualids top_scale_ids;

   /*std::vector<std::pair<size_t, float>> scales;
    scales.reserve(static_cast<size_t>(keypoints->size()));
    for (size_t i = 0; i < keypoints->size(); ++i) {
        scales.emplace_back(i, (*keypoints)[i].ComputeScale());
    }

    std::partial_sort(scales.begin(), scales.begin() + num_features,
        scales.end(),
        [](const std::pair<size_t, float> scale1,
            const std::pair<size_t, float> scale2) {
                return scale1.second > scale2.second;
        });*/

    top_scale_keypoints.reserve(num_features);
    top_scale_ids.resize(num_features, ids->ids.cols());
    int top_scale_num_feats = 0;
    for (int i = 0; i < num_features; i++) {
        int kp_index = (*keypoints).size() - num_features+i;
        top_scale_keypoints.push_back((*keypoints)[kp_index]);
        top_scale_ids(i, 1) = ids->ids(kp_index, 1);
        top_scale_ids(i, 0) = i;
    }
    /*for (size_t i = 0; i < num_features; ++i) {
        featRes.push_back(scales[i].first);
        top_scale_keypoints[i] = (*keypoints)[scales[i].first];
        top_scale_ids.row(i) = ids->ids.row(scales[i].first);
    }*/

    *keypoints = top_scale_keypoints;
    ids->ids = top_scale_ids;   
}

void helper::ExtractTopDescriptors(colmap::FeatureKeypoints* keypoints, colmap::FeatureDescriptors* descriptors, const size_t num_features) {

    if (keypoints->size() != descriptors->rows()) {
        throw std::invalid_argument("Error: visual ids and keypoints are not in the same length.");
    }
    if (num_features == 0) {
        throw std::invalid_argument("Error: num of features need to be non-zero");
    }

    if (static_cast<size_t>(descriptors->rows()) <= num_features || num_features == -1) {
        return;
    }

    colmap::FeatureKeypoints top_scale_keypoints;
    colmap::FeatureDescriptors top_scale_descriptors;

    /*std::vector<std::pair<size_t, float>> scales;
     scales.reserve(static_cast<size_t>(keypoints->size()));
     for (size_t i = 0; i < keypoints->size(); ++i) {
         scales.emplace_back(i, (*keypoints)[i].ComputeScale());
     }

     std::partial_sort(scales.begin(), scales.begin() + num_features,
         scales.end(),
         [](const std::pair<size_t, float> scale1,
             const std::pair<size_t, float> scale2) {
                 return scale1.second > scale2.second;
         });*/

    top_scale_keypoints.reserve(num_features);
    top_scale_descriptors.resize(num_features, descriptors->cols()); // number of sift descriptor digit
    int top_scale_num_feats = 0;
    for (int i = 0; i < num_features; i++) {
        int kp_index = (*keypoints).size() - num_features + i;
        top_scale_keypoints.push_back((*keypoints)[kp_index]);
        top_scale_descriptors.row(i) = (*descriptors).row(kp_index);
    }
    /*for (size_t i = 0; i < num_features; ++i) {
        featRes.push_back(scales[i].first);
        top_scale_keypoints[i] = (*keypoints)[scales[i].first];
        top_scale_ids.row(i) = ids->ids.row(scales[i].first);
    }*/

    *keypoints = top_scale_keypoints;
    *descriptors = top_scale_descriptors;
}

std::unordered_set<int> helper::pickSet(int N, int k, std::mt19937& gen)
{
    std::unordered_set<int> elems;
    for (int r = N - k; r < N; ++r) {
        int v = std::uniform_int_distribution<>(1, r)(gen);

        // there are two cases.
        // v is not in candidates ==> add it
        // v is in candidates ==> well, r is definitely not, because
        // this is the first iteration in the loop that we could've
        // picked something that big.

        if (!elems.insert(v).second) {
            elems.insert(r);
        }
    }
    return elems;
}
