#include "fileop.h"
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include "constAndTypes.h"
#include "helper.h"

using ArgList = fileop::ArgList;
using ArgType = constandtypes::ArgType;
using namespace cv;
using namespace std;
namespace fs = std::filesystem;

const char* keys =
"{ help h |                  | Print help message. }"
"{ tool   |      both        | Lib used for SIFT, OpenCV or VLFeat, default both. }"
"{ path   |                  | Path to the training image folder, learning visual words, not compatable with input1/2 }"
"{ img    |                  | Path to single test input img, computing and visualize the keypoints and descriptor for the img }"
"{ input1 |                  | Image matching pairs, Path to input image 1, not comp. with path }"
"{ input2 |                  | Image matching pairs, Path to input image 2, not comp. with path }";

void fileop::write_to_file(std::string name, std::vector<KeyPoint>& kpts, Mat& kCenters) {
    if (!fs::exists("Result")) {
        fs::create_directories("Result");
    }
    std::ofstream CSVOutput;
    int nKpts = kpts.size();
    CSVOutput.open(std::string("Result/" + name + "_statistics_" + std::to_string(nKpts) + "_" + dateTime() + ".csv"), std::fstream::out | std::fstream::app);

    //keypoint write to file
    if (!kpts.empty()) {
        //input stream for headers
        CSVOutput << "Oritentation" << "," << "Octave" << "," << "layer" << "," << "pt.x" << "," << "pt.y" << "," << "scale" << "\n";

        //write data to the file
        int numOkpt = kpts.size();
        int max_noctave = 0;
        for (int i = 0; i < numOkpt; i++) {
            auto kp = kpts[i];
            int noctave, nlayer;
            float vscale;
            unpackOctave(kp, noctave, nlayer, vscale);
            if (noctave > max_noctave) {
                max_noctave = noctave;
            }

            //unpack keypoints value
            CSVOutput << kp.angle << ","
                << noctave << ","
                << nlayer << ","
                << kp.pt.x << ","
                << kp.pt.y << ","
                << vscale << "\n";
        }
        CSVOutput.close();
        cout << "->keypoints store finish with octave num: " << max_noctave << endl;
    }

    //write kcenters to files
    if (!kCenters.empty()) {
        cv::FileStorage filewriter("Result/" + name + "_kmeansCenter.yml", cv::FileStorage::WRITE);
        filewriter << "kcenters" << kCenters;
        cout << "->visual words are written to file kmeansCenter.yml......" << endl;
    }
}

 ArgList fileop::funTestRead(int argc, const char* argv[], std::vector<std::string>& trainFilePaths, std::vector<std::string>& testFilePaths) {

    //open files with link of images
    CommandLineParser parser(argc, argv, keys);
    ArgList cmd; //add arg options to this variables

    /*Mat img1 = imread(samples::findFile(parser.get<String>("input1"), false), IMREAD_GRAYSCALE);
    Mat img2 = imread(samples::findFile(parser.get<String>("input2"), false), IMREAD_GRAYSCALE);*/
    /**
     * if path is provided, we compute the correspondence
     *
     * */
    if (argc > 3) {
        throw std::invalid_argument("provides too much arguments! please check -h for details");
    }
    /**if (argc == 2 && img1.empty() || img2.empty()) {
        throw std::invalid_argument("please specify only either -path or -img!");

    }**/
    if (parser.has("tool")) {
        String tool = parser.get<String>("tool");
        if (tool == "vlfeat") {
            cmd.tool = ArgType::TOOL_VLFEAT;
        }
        else if (tool == "opencv")
        {
            cmd.tool = ArgType::TOOL_OPENCV;
        }
        else if (tool == "both") {
            cmd.tool = ArgType::TOOL_OPENCV_AND_VLFEAT;
        }
        else {
            throw std::invalid_argument("-tool= " + tool + " is not supported");
        }
    }

    if (parser.has("path") && argc == 2) {    //if (!imgs_path.empty() && img1.empty() && img2.empty() && img.empty()) {
        fs::path imgs_path = parser.get<String>("path");
        if (!imgs_path.empty() && !fs::exists(imgs_path)) {
            cout << "ERROR: provided imgs path doesn't exist!" << endl;

        }
        cmd.mode = ArgType::MODE_TRAIN;

        //find the train imgs
        fs::path train_path = imgs_path;
        fs::path test_path = imgs_path;
        train_path /= "train";
        test_path /= "test";
        cout << "Current training images path: " << train_path << endl;
        cout << "Current testing images path: " << test_path << endl;
        if (!fs::exists(train_path)) {
            throw std::invalid_argument("train subfolder of provided path doesn't exist!");
        }
        else if (!fs::exists(train_path)) {
            std::cerr << "WARNING: test subfolder for provided path doesn't exist! read training imgs from training path......" << endl;
        }
        else {
            for (const auto& entry : fs::directory_iterator(test_path)) {
                std::string::size_type idx;
                idx = entry.path().string().rfind('.');
                if (idx != std::string::npos)
                {
                    std::string extension = entry.path().string().substr(idx + 1);
                    if (extension == "jpg") {
                        testFilePaths.push_back(entry.path().string());
                        cout << "test img is added and found at: " << entry.path().string() << "......" << endl;
                    }
                    else {
                        cout << "img " + entry.path().string() + ": Extension" + extension + " is not supported dismiss the image" << endl;
                    }
                }
            }
        }

        for (const auto& entry : fs::directory_iterator(train_path)) {
            std::string::size_type idx;
            idx = entry.path().string().rfind('.');
            if (idx != std::string::npos)
            {
                std::string extension = entry.path().string().substr(idx + 1);
                if (extension == "jpg") {
                    trainFilePaths.push_back(entry.path().string());
                    cout << "img is added and found at: " << entry.path().string() << "......" << endl;
                }
                else {
                    cout << "img " + entry.path().string() + ": Extension" + extension + " is not supported dismiss the image" << endl;
                }
            }
        }
        return cmd;
    }
    else if (parser.has("img") && argc == 2) {    //if (imgs_path.empty() && img1.empty() && img2.empty() && !img.empty()) {
        fs::path img = parser.get<String>("img");
        if (!fs::exists(img)) {
            throw std::invalid_argument("provided -img path is not a valid image.");
        }

        cmd.mode = ArgType::MODE_DEMO;

        cout << "Current input image: " << img << endl;
        std::string::size_type idx;
        idx = img.string().rfind('.');
        if (idx != std::string::npos)
        {
            std::string extension = img.string().substr(idx + 1);
            if (extension == "jpg") {
                trainFilePaths.push_back(img.string());
                cout << "img is added and found at: " << img.string() << "......" << endl;
            }
            else {
                throw std::invalid_argument("input img type " + extension + "is not supported");
            }
        }
        return cmd;
    }
    else if (parser.has("input1") && parser.has("input2") && argc == 3)//
    {
        fs::path img1 = parser.get<String>("input1");
        fs::path img2 = parser.get<String>("input2");
        if (!fs::exists(img1) || !fs::exists(img2)) {
            throw std::invalid_argument("provided -img1/-img2 path is not a valid image.");
        }
        cmd.mode=ArgType::MODE_MATCHING;

        cout << "Two input image path is: \n ->img1: " << img1.string() << "\n ->img2: " << img2.string() << endl;
        std::string::size_type idx1, idx2;
        idx1 = img1.string().rfind('.');
        idx2 = img2.string().rfind('.');
        if (idx1 != std::string::npos && idx2 != std::string::npos)
        {
            std::string extension1 = img1.string().substr(idx1 + 1);
            std::string extension2 = img2.string().substr(idx2 + 1);
            if (extension1 == "jpg" && extension2 == "jpg") {
                trainFilePaths.push_back(img1.string());
                trainFilePaths.push_back(img2.string());
            }
            else {
                throw std::invalid_argument("input img type " + extension1 + "is not supported");
            }
        }
        return cmd;
    }
    else {
        throw std::invalid_argument("Input Arguments are not supported, check your args / executables!");
    }

    //TODO: function for read other types of arg inputs
}
