#include <opencv2/core.hpp>
#include <iostream>
#include "cluster.h"
#include "constAndTypes.h"
extern "C" {
    #include "vl/generic.h"
    #include "vl/kmeans.h"
    #include "vl/kdtree.h"
}
using namespace std;
namespace cAt = constandtypes;

void cluster::openCV_visual_words_compute(cv::Mat& allDescripts, cv::Mat& kCenters) {
    clock_t sTime = clock();
    cout << "start k-means learning..." << endl;
    cv::Mat labels; //stores the trained labels
    //k-means 
    kmeans(allDescripts, cAt::centers, labels, cAt::criteria, cAt::numOfAttemp, cv::KMEANS_PP_CENTERS, kCenters);
    cout << "->successfully produce cluster centers MAT with size: " << kCenters.rows << endl;
    cout << "-> kmeans learning spent " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " sec......" << endl;
}
void cluster::vl_visual_word_compute(cv::Mat& allDescrip, cv::Mat& kCenters) {
    clock_t sTime = clock();
    cout << "start k-means learning with " << cAt::centers << " centers..." << endl;
    //all kmeans parameters setting corresponding with OpenCV setting
    int dim = 128;
    int numOfpts = allDescrip.rows;
    double energy;
    VlKMeans* km = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
    vl_kmeans_set_algorithm(km, VlKMeansLloyd);

    //data row major
    float* data = allDescrip.ptr<float>(0);
    vl_kmeans_init_centers_plus_plus(km, data, dim, numOfpts, cAt::centers);

    vl_kmeans_set_num_repetitions(km, 3);

    /**The relative energy variation is calculated after the $t$ - th update
    ** to the parameters as :
    **
    **  \[\epsilon_t = \frac{ E_{t - 1} -E_t }{E_0 - E_t} \]
    **  set the relative energy variation to the initial minus current energy as the stop criteria
    **  OpenCV uses the absolute variation of pixel corner position for kcenters as stop criteria, while VLFeat is the relative energy variation.
    **/
    vl_kmeans_set_min_energy_variation(km, cAt::accuracy);
    vl_kmeans_set_max_num_iterations(km, cAt::numOfItera);
    energy = vl_kmeans_refine_centers(km, data, numOfpts);

    //obtain kcenters result
    const float* center_ptr = (float*)vl_kmeans_get_centers(km);

    //iterate and stores the kcenters as cv::Mat
    cv::Mat1f centerDescrip(1, dim);
    for (int i = 0; i < cAt::centers; i++) {
        for (int j = 0; j < dim; j++) {
            centerDescrip(0, j) = center_ptr[i * dim + j];
        }
        kCenters.push_back(centerDescrip);

    }
    cout << "-> VLFeat kmeans learning spent " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " sec" << endl;
}