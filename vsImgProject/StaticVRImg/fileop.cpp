#include "fileManager.h"
#include <iostream>
#include <filesystem>
#include <fstream>
#include "helper.h"
#include <igraph.h>

using ArgList = fileManager::ArgList;
using ArgType = fileManager::ArgType;
using namespace cv;
using namespace std;



//parameters definitions
int fileManager::parameters::octave = -1; //default to -1 to compute every possible octave: log2(min(width,height))
int fileManager::parameters::noctaveLayer = 3; // scale layers per octave
int fileManager::parameters::firstOctaveInd = -1; // learning start from 1th octave, -1 for more details
double fileManager::parameters::sigma_0 = 1.6; // sigma for the #0 octave
int fileManager::parameters::centers = 200;    // k-means center detection, defines the number of centers
int fileManager::parameters::numOfAttemp = 5; //times of try to compute the center for each cluster, five times to choose the best one
int fileManager::parameters::numOfItera = 20;
int fileManager::parameters::descriptDim = 128;
double fileManager::parameters::accuracy = 1e-3;
double fileManager::parameters::siftEdgeThres = 10; // sift paper setting
double fileManager::parameters::siftPeakThres = 0.03; // sift paper setting
double fileManager::parameters::imgScale = 0.5; //image scaling during detection and drawing

//OpenCV relevent setting
TermCriteria fileManager::parameters::criteria = TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, numOfItera, accuracy); //stop criteria, COUNT means number of iter, EPS means convergence accuracy
float fileManager::parameters::MATCH_THRES = 0.7; //define the threshold for matching 

//matching setting
int fileManager::parameters::maxNumComp = 200; //sift paper setting
int fileManager::parameters::numOfNN = 2; //sift paper setting
double fileManager::parameters::distRat = 0.8; // sift paper setting for threshold

//graph building relevent setting
size_t fileManager::parameters::maxNumDeg = 5; //
double fileManager::parameters::radDegLim = std::numeric_limits<double>::infinity(); //default infinity


void fileManager::write_to_file(std::string name, std::vector<KeyPoint>& kpts, Mat& kCenters) {
    if (!fs::exists("Result")) {
        fs::create_directories("Result");
    }
    

    //keypoint write to file
    if (!kpts.empty()) {
        std::ofstream CSVOutput;
        int nKpts = kpts.size();
        CSVOutput.open(std::string("Result/" + name + "_statistics_" + std::to_string(nKpts) + "_" + dateTime() + ".csv"), std::fstream::out | std::fstream::app);
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
        cout << "->keypoints store finished with octave num: " << max_noctave << endl;
    }

    //write kcenters to files
    if (!kCenters.empty()) {
        cv::FileStorage filewriter("Result/" + name + "_kmeansCenter.yml", cv::FileStorage::WRITE);
        filewriter << "kcenters" << kCenters;
        cout << "->visual words are written to file "<<name<<"_kmeansCenter.yml......" << endl;
    }
}


/**
* This function is specified for project FuntestVRN file read
* argc, argv: main function argument list
* trainFilePaths, testFilePaths: resultant files path 
* keys: keys for the file read
*/
 ArgList fileManager::funTestRead(int argc, const char* argv[], std::vector<std::string>& trainFilePaths, std::vector<std::string>& testFilePaths, const char* keys) {

    //parser argument list
    CommandLineParser parser(argc, argv, keys);
    if (!parser.check()) {
        throw std::invalid_argument("argument list error! check your inputs!");
    }
    ArgList cmd; //add arg options to this variables

    /*Mat img1 = imread(samples::findFile(parser.get<String>("input1"), false), IMREAD_GRAYSCALE);
    Mat img2 = imread(samples::findFile(parser.get<String>("input2"), false), IMREAD_GRAYSCALE);*/
    /**
     * if path is provided, we compute the correspondence
     *
     * */
   /* if (argc > 4) {
        throw std::invalid_argument("provided arguments out of range! please check -h for details");
    }*/
    /**if (argc == 2 && img1.empty() || img2.empty()) {
        throw std::invalid_argument("please specify only either -path or -img!");

    }**/
   
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
    
    String mode = parser.get<String>("mode");

    if (mode=="train" && parser.has("path")) {    //if (!imgs_path.empty() && img1.empty() && img2.empty() && img.empty()) {
        fs::path imgs_path = parser.get<String>("path");
        if (!imgs_path.empty() && !fs::exists(imgs_path)) {
            cout << "Mode Train: ERROR: provided imgs path doesn't exist!" << endl;

        }
        cmd.mode = ArgType::MODE_TRAIN;

        //find the train imgs
        fs::path train_path = imgs_path;
        fs::path test_path = imgs_path;
        train_path /= "train";
        test_path /= "test";
        cout << "Mode Train: Current training datasets images path: " << train_path << endl;
        cout << "Mode Train: Current testing datasets images path: " << test_path << endl;
        if (!fs::exists(train_path)) {
            throw std::invalid_argument("Mode Train: Training subfolder of provided path doesn't exist!");
        }
        else if (!fs::exists(train_path)) {
            std::cerr << "Mode Train: WARNING: test subfolder for provided path doesn't exist! read training imgs from training path......" << endl;
        }
        else {
            if (fs::exists(test_path)) {
                for (const auto& entry : fs::directory_iterator(test_path)) {
                    std::string::size_type idx;
                    idx = entry.path().string().rfind('.');
                    if (idx != std::string::npos)
                    {
                        std::string extension = entry.path().string().substr(idx + 1);
                        if (extension == "jpg" || extension == "JPG" || extension == "JPEG") {
                            testFilePaths.push_back(entry.path().string());
                            cout << "Mode Train: Testing img is added and found at: " << entry.path().string() << "......" << endl;
                        }
                        else {
                            cout << "Mode Train: img " + entry.path().string() + ": Extension" + extension + " is not supported ignore the image" << endl;
                        }
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
                if (extension == "jpg" || extension == "JPG" || extension == "JPEG") {
                    trainFilePaths.push_back(entry.path().string());
                    cout << "Mode Train: img is added and found at: " << entry.path().string() << "......" << endl;
                }
                else {
                    cout << "Mode Train: img " + entry.path().string() + ": Extension " + extension + " is not supported dismiss the image" << endl;
                }
            }
        }
        return cmd;
    }
    else if (mode=="demo") {   
        fs::path img = parser.get<String>("path");
        if (img.empty() || !fs::exists(img)) {
            throw std::invalid_argument("Mode Demo: ERROR: provided path is not a valid image.");
        }

        cmd.mode = ArgType::MODE_DEMO;

        cout << "Mode Demo: Current input image: " << img << endl;
        std::string::size_type idx;
        idx = img.string().rfind('.');
        if (idx != std::string::npos)
        {
            std::string extension = img.string().substr(idx + 1);
            if (extension == "jpg" || extension == "JPG" || extension == "JPEG") {
                trainFilePaths.push_back(img.string());
                cout << "Mode Demo: img is added and found at: " << img.string() << "......" << endl;
            }
            else {
                throw std::invalid_argument("Mode Demo: input img type " + extension + "is not supported");
            }
        }
        return cmd;
    }
    else if (mode=="matching")
    {
        fs::path imgs = parser.get<String>("path");
        if (!fs::exists(imgs)) {
            throw std::invalid_argument("Mode Matching: provided path is not a valid path.");
        }
        cmd.mode=ArgType::MODE_MATCHING;
        for (const auto& entry : fs::directory_iterator(imgs)) {
            std::string::size_type idx;
            idx = entry.path().string().rfind('.');
            if (idx != std::string::npos)
            {                
                trainFilePaths.push_back(entry.path().string());
                cout << "Mode Matching: img is added and found at: " << entry.path().string() << "......" << endl;                                
            }
        }
        if (trainFilePaths.size() != 2) {
            throw std::invalid_argument("Mode Matching: ERROR: provided path comtains incorrect number of imgs for matching (should be 2).");
        }

        cout << "Two input image paths are: \n ->img1: " << trainFilePaths[0] << "\n ->img2: " << trainFilePaths[1] << endl;
        return cmd;
    }
    else {
        throw std::invalid_argument("Input Arguments are not supported, check your args list.");
    }

    //TODO: function for read other types of arg inputs
}

//warning handler that disable warning when write to graphML
void null_warning_handler(const char* reason, const char* file, int line, int igraph_errno) {

}

void fileManager::write_graph(igraph_t& graph, string name, string mode) {
    if (!fs::exists("Result")) {
        fs::create_directories("Result");
    }
    std::string fileName = name;
    if (mode == "graphml") {
        FILE* graph_writer = fopen(("Result/"+fileName + ".graphml").c_str(), "w");
        igraph_warning_handler_t *warning;
        warning = igraph_set_warning_handler(null_warning_handler);
        igraph_write_graph_graphml(&graph, graph_writer, true);
        igraph_set_warning_handler(warning);
    }
    else {
        std::cout << "write graph: unsupported graph saving mode" << std::endl;
        throw std::invalid_argument("unsupported saving mode");
    }
}

void fileManager::read_user_set(fs::path& params) {

    //read path
    json jsonlist;
    if (!fs::exists(params)) {
        cout << "Read Params: provided user setting .json doesn't exist!" << endl;
        throw std::invalid_argument("PATH_NOT_EXIST");
    }

    ifstream f(params.c_str());
    if (!f) {
        std::cout << "Read Parameters: file read failed" << endl;
        throw std::runtime_error("READ_FAILED");
    }
    //check the status, throw exception for error
    f >> jsonlist;
    fileManager::parameters::octave = jsonlist.value("octave", fileManager::parameters::octave);
    fileManager::parameters::noctaveLayer = jsonlist.value("noctaveLayer", fileManager::parameters::noctaveLayer);
    fileManager::parameters::firstOctaveInd = jsonlist.value("firstOctaveInd", fileManager::parameters::firstOctaveInd);
    fileManager::parameters::sigma_0 = jsonlist.value("sigma_0", fileManager::parameters::sigma_0);
    fileManager::parameters::centers = jsonlist.value("centers", fileManager::parameters::centers);
    fileManager::parameters::numOfAttemp = jsonlist.value("numOfAttemp", fileManager::parameters::numOfAttemp);
    fileManager::parameters::numOfItera = jsonlist.value("numOfItera", fileManager::parameters::numOfItera);
    fileManager::parameters::accuracy = jsonlist.value("accuracy", fileManager::parameters::accuracy);
    fileManager::parameters::MATCH_THRES = jsonlist.value("MATCH_THRES", fileManager::parameters::MATCH_THRES);
    fileManager::parameters::numOfNN = jsonlist.value("numOfNN", fileManager::parameters::numOfNN);
    fileManager::parameters::maxNumDeg = jsonlist.value("maxNumDeg", fileManager::parameters::maxNumDeg);
    int radLim = jsonlist.value("radDegLim", fileManager::parameters::radDegLim);
    if (radLim != -1) {
        fileManager::parameters::radDegLim = radLim;
    }
    fileManager::parameters::imgScale = jsonlist.value("imgScale", fileManager::parameters::imgScale);
    fileManager::parameters::siftEdgeThres = jsonlist.value("siftEdgeThres", fileManager::parameters::siftEdgeThres);
    fileManager::parameters::siftPeakThres = jsonlist.value("siftPeakThres", fileManager::parameters::siftPeakThres);
}
