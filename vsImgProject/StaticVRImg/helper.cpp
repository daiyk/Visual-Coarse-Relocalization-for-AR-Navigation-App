#include "helper.h"
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
