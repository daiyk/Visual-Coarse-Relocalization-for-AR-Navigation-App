#include <opencv2/core.hpp>
#include <iostream>
#include "cluster.h"
#include "fileManager.h"
extern "C" {
    #include "vl/generic.h"
    #include "vl/kmeans.h"
    #include "vl/kdtree.h"
}
using namespace std;
using params = fileManager::parameters;

void cluster::openCV_visual_words_compute(cv::Mat& allDescripts, cv::Mat& kCenters) {
    clock_t sTime = clock();
    cout << "start k-means learning..." << endl;
    cv::Mat labels; //stores the trained labels
    //k-means
    //check the kmeans setting
    if (params::centers > allDescripts.rows) {
        std::cout << "Kmeans: centers number exceeds the descriptor's number reset the centers to " << allDescripts.rows << " centers instead" << endl;
        params::centers = allDescripts.rows;
    }
    kmeans(allDescripts, params::centers, labels, params::criteria, params::numOfAttemp, cv::KMEANS_PP_CENTERS, kCenters);
    cout << "->successfully produce cluster centers MAT with size: " << kCenters.rows << endl;
    cout << "-> kmeans learning spent " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " sec......" << endl;
}
void cluster::vl_visual_word_compute(cv::Mat& allDescrip, cv::Mat& kCenters, int centers) {
    clock_t sTime = clock();
    cout << "start k-means learning with " << centers << " centers..." << endl;
    //all kmeans parameters setting corresponding with OpenCV setting
    int dim = 128;
    size_t numOfpts = allDescrip.rows;
    double energy;
    vl_set_num_threads(0);
    VlKMeans* km = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
    vl_kmeans_set_algorithm(km, VlKMeansANN);

    //data row major
    float* data = allDescrip.ptr<float>(0);
    //check kmeans setting
    if (centers > allDescrip.rows) {
        std::cout << "Kmeans: centers number exceeds the descriptor's number reset the No.centers to " << allDescrip.rows << " instead" << endl;
        centers = allDescrip.rows;
    }
    vl_kmeans_init_centers_plus_plus(km, data, dim, numOfpts, centers);

    vl_kmeans_set_max_num_iterations(km, 100);
    /*vl_kmeans_set_num_repetitions(km, params::numOfAttemp);*/

    /**The relative energy variation is calculated after the $t$ - th update
    ** to the parameters as :
    **
    **  \[\epsilon_t = \frac{ E_{t - 1} -E_t }{E_0 - E_t} \]
    **  set the relative energy variation to the initial minus current energy as the stop criteria
    **  OpenCV uses the absolute variation of pixel corner position for kcenters as stop criteria, while VLFeat is the relative energy variation.
    **/
    /*vl_kmeans_set_min_energy_variation(km, params::accuracy);
    vl_kmeans_set_max_num_iterations(km, params::numOfItera);*/
    vl_kmeans_refine_centers(km, data, numOfpts);

    //obtain energy
    energy = vl_kmeans_get_energy(km);
    //obtain kcenters result
    const float* center_ptr = (float*)vl_kmeans_get_centers(km);

    //iterate and stores the kcenters as cv::Mat
    cv::Mat1f centerDescrip(1, dim);
    for (int i = 0; i < centers; i++) {
        for (int j = 0; j < dim; j++) {
            centerDescrip(0, j) = center_ptr[i * dim + j];
        }
        kCenters.push_back(centerDescrip);

    }
    cout << "-> VLFeat kmeans learning spent " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " sec" << endl;
}
