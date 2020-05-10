#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "StaticVRImg/matcher.h"
#include <stdio.h>
using namespace cv;
namespace fs = std::filesystem;
const char* keys =
"{ help h |                  | Print help message. }"
"{ data   |                  | path to the dataset folder for descriptor computation. }";




int main(int argc, char* argv[]) {
	CommandLineParser parser(argc, argv, keys);
	fs::path descrip_foldName("VRNHpatches");
	std::ofstream metricsOpen;
	matcher::matchOut matchRel;
	if (!fs::exists(descrip_foldName)) {
		fs::create_directories(descrip_foldName);
	}

	metricsOpen.open(descrip_foldName/"metrics.csv");
	if(!parser.has("data")){
		std::cout << " -> Please specify the image folder path" << std::endl;
		return 0;
	}
	std::string f = parser.get<std::string>("data");
	std::vector<std::string> subf;
	if (!fs::is_directory(f)) {
		std::cout << " -> Provided path is not a directory!" << std::endl;
		return 0;
	}
	//iterate through all subdirectories
	for (const auto& entry : fs::directory_iterator(f)) {
		subf.push_back(entry.path().string());
	}

	//iterate each subfolder and add to training path
	for (auto subPath : subf) {
		std::cout << "Processing path: " << subPath << std::endl;
		fs::path filename(subPath);
		if (filename.has_filename()&&fs::is_directory(filename))
		{
			fs::path storeFolder = filename.filename();
			fs::create_directories(descrip_foldName/storeFolder);
			std::vector<std::string> imgs;
			for (auto entry : fs::directory_iterator(filename)) {
				//process the image and write out the descriptors
				if(entry.path().extension()==".ppm")
					imgs.push_back(entry.path().string());
			}

			//do matching for each img w.r.t the first as reference
			std::string& refImg = imgs[0];

			for (int i = 0; i < imgs.size();i++) {
				matchRel=matcher::kdTreeDemo(refImg, imgs[i], false);
				Mat outImg;
				cv::drawMatches(imread(imgs[i]), matchRel.source, imread(imgs[0]), matchRel.refer, matchRel.matches, outImg, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

				imwrite((descrip_foldName/storeFolder/(std::to_string(i)+"toRef.jpg")).string(), outImg);
				//write the simple metric data
				float metricDist = 0;
				for (int j = 0; j < matchRel.matches.size(); j++)
				{
					metricDist += matchRel.matches[i].distance / matchRel.matches.size();
				}
				metricsOpen << (filename.filename().string() + (std::to_string(i) + "toRef.jpg")) << "," << metricDist << "\n";
				std::cout << imgs[i] << " is processed" << std::endl;
			}
			//open file for writ
		}
		else {
			std::cout << filename.string() + " is not a valid directory!" << std::endl;
		}

		//create directory for the resultant descriptors
	}
	metricsOpen.close();

}