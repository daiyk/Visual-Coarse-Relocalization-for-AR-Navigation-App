#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "Shlwapi.h"
#include "omp.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
extern "C" {
  #include "vl/generic.h"
  #include <vl/array.h>
  #include <vl/covdet.h>
  #include <vl/sift.h>
  #include <vl/fisher.h>
}
#include <iostream>
using namespace cv;
namespace fs = std::filesystem;
using std::cout;
using std::endl;
// const char *keys =
//     "{ help h |                  | Print help message. }"
//     "{ path   |                  | Path to the image folder, not comp. with input1/2 }"
//     "{ input1 |                  | Path to input image 1, not comp. with path }"
//     "{ input2 |                  | Path to input image 2, not comp. with path }";
extern "C" {
  #include "vl/vlad.h"
  #include "vl/sift.h"
  #include "vl/generic.h"
}
void vlimg_descips_compute(std::vector<String> &paths, Mat &kCenters)
{
    size_t num_imgs = paths.size();
    Mat allDescrips;
    if(num_imgs==0){; 
        throw std::invalid_argument("imgs folder is empty!");
    }
    omp_set_num_threads(6);
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for(size_t i=0;i<num_imgs;i=i+5){
            // int minHessian = 400; // for SURF detector only
            Mat grayImg;
            Mat grayImgFl;
            Mat imgDescrips;
            if(!fs::exists(fs::path(paths[i]))){
                cout<<"Warning: "<<paths[i]<<"does not exist, pass the image"<<endl;
                continue;
            }
            cvtColor(imread(paths[i]),grayImg,COLOR_BGR2GRAY);

            //surf to detect and compute
            size_t width = grayImg.size().width;
            size_t height = grayImg.size().height;
            
            int octave =3;
            int noctaveLayer = 3;
            int octave_start = 1;
            double sigma_0 = 1.6; //sigma for the #0 octave

            auto vl_sift = vl_sift_new(width,height,octave,noctaveLayer,1);//define vl sift processor
            vl_sift_set_edge_thresh(vl_sift,10);
            vl_sift_set_peak_thresh(vl_sift,0.04);

            grayImg.convertTo(grayImgFl,CV_32F,1.0/255.0);
            
            float *img_ptr = grayImgFl.ptr<float>();
            
            if(!grayImgFl.isContinuous()){
                std::cerr<<"ERROR: when read "<<paths[i]<<" OpenCV finds uncontinuous address"<<endl;
                continue;
            }
            
            //go trough the loop of gaussian pyramid
            int result = vl_sift_process_first_octave(vl_sift,img_ptr);
            Mat1f allDescriptors;
            while(result!=VL_ERR_EOF){
                vl_sift_detect(vl_sift);

                //to get each keypoints
                int keyPtsNum = vl_sift_get_nkeypoints(vl_sift);

                const auto *keypoints = vl_sift_get_keypoints(vl_sift);

                //loop each keypoints and get orientation
                for(int i=0;i<keyPtsNum;i++){
                    double rot[4];
                    int nOrit = vl_sift_calc_keypoint_orientations(vl_sift,rot,&keypoints[i]);
                    //get the descriptors for each computed orientation
                    for(int j = 0;j<nOrit;j++){
                        float *curr_descriptor;
                        vl_sift_calc_keypoint_descriptor(vl_sift,curr_descriptor,&keypoints[i],rot[j]);
                        imgDescrips.push_back(Mat1f(1,128,curr_descriptor));
                    }
                    
                }
                result = vl_sift_process_next_octave(vl_sift);
            }
            
            //iterate through the 

            //define vlfeat 
            //for each octave to calculate the scale space
            //add to the total statistics, descriptors and keypoints
            #pragma omp critical
            {
                allDescrips.push_back(imgDescrips);
            }
         //delete sift
         vl_sift_delete(vl_sift);   
        }
    }   
    clock_t sTime=clock();
    cout<<"start k-means learning..."<<endl;
}


int main (int argc, const char * argv[]) {
  VL_PRINT ("Hello world!") ;

  std::cout<<endl<<"test success"<<std::endl;
  return 0;
}