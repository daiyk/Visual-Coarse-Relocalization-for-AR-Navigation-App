#include "fileManager.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "helper.h"
#include <igraph.h>

using ArgList = fileManager::ArgList;
using ArgType = fileManager::ArgType;
using namespace cv;
using namespace std;

//default user set location
fs::path fileManager::user_set_default = "D:\\thesis\\Visual-Coarse-Relocalization-for-AR-Navigation-App\\User\\vrn_set.json";

//parameters definitions
string fileManager::parameters::userSetPath = ""; //need to read this from the user set file
int fileManager::parameters::octave = -1; //default to -1 to compute every possible octave: log2(min(width,height))
int fileManager::parameters::noctaveLayer = 3; // scale layers per octave
int fileManager::parameters::firstOctaveInd = -1; // learning start from 1th octave, -1 for more details
double fileManager::parameters::sigma_0 = 1.6; // sigma for the #0 octave
int fileManager::parameters::centers = 200;    // k-means center detection, defines the number of centers
int fileManager::parameters::numOfAttemp = 5; //times of try to compute the center for each cluster, five times to choose the best one
int fileManager::parameters::numOfItera = 20; 
int fileManager::parameters::descriptDim = 128;
double fileManager::parameters::accuracy = 1e-3;
double fileManager::parameters::siftEdgeThres = 10.0; // sift paper setting
double fileManager::parameters::siftPeakThres = 0.02 / fileManager::parameters::noctaveLayer; // sift paper setting
double fileManager::parameters::imgScale = 1.0; // image scaling during detection and drawing
int fileManager::parameters::maxNumOrient = 4; // max number orientation extracted by vlfeat covdet feature detector, default 4 for SIFT
int fileManager::parameters::maxNumFeatures = -1;// max number of features allowed in the detection
string fileManager::parameters::tfidfPath = ""; // path to the tfidf file 
//OpenCV relevent setting
TermCriteria fileManager::parameters::criteria = TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, numOfItera, accuracy); //stop criteria, COUNT means number of iter, EPS means convergence accuracy


//matching setting
int fileManager::parameters::maxNumComp = 200; //sift paper setting
int fileManager::parameters::numOfNN = 2; //sift paper setting
float fileManager::parameters::MATCH_THRES = 0.7; //define the threshold for matching 

//vlad setting
int fileManager::parameters::vladCenters = 16; //default vlad centers

//graph building relevent setting
size_t fileManager::parameters::maxNumDeg = 5; //
double fileManager::parameters::radDegLim = std::numeric_limits<double>::infinity(); //default infinity

//test relative setting
int fileManager::parameters::sampleSize = 0;
int fileManager::parameters::imgsetSize = 0;

//covismap relavent setting
double fileManager::parameters::PCliques = 0.05;
double fileManager::parameters::PCommonwords = 0.04;

/*Args:
*   paths: must be empty string vector, the scaned file paths are stored here
*   path: string of path to the folder for scan files
*/
void fileManager::read_files_in_path(std::string path, std::vector<std::string> &paths, std::vector<std::string> extensions){
    fs::path imgs_path = path;
    if (!imgs_path.empty() && !fs::exists(imgs_path)) {
        cout << "read_files_in_path: provided root path doesn't exist!" << endl;
        return;
    }
    paths.clear();
    for (const auto& entry : fs::directory_iterator(imgs_path)) {
        std::string::size_type idx;
        idx = entry.path().string().rfind('.');
        if (idx != std::string::npos)
        {
            std::string extension = entry.path().string().substr(idx + 1);
            if (!extensions.empty() && std::find(extensions.begin(),extensions.end(),extension)!=extensions.end()) {
                paths.push_back(entry.path().string());
                cout << "Files in Path: file is added with: " << entry.path().string() << "......" << endl;
            }
            else
            {
                cout << "Files in Path: file: " << entry.path() << " is not with the allowed extensions, ignore the file......" << endl;
            }
           
            
        }
    }

}

Eigen::MatrixXd fileManager::read_point3Ds(std::string file_path,int num_images) {
    std::ifstream input(file_path,std::ifstream::in);

    //loop over the file and read the point3D
    std::string line;
    int count = 0;
    Eigen::MatrixXd overlap_matrix = Eigen::MatrixXd::Zero(num_images+1, num_images+1);
    while (std::getline(input, line)) {
        if (count < 3)//ignore the first three commentlines
        {
            count++;
            continue;
        }
        std::istringstream subinput(line);
        std::vector<int> imageIds;
        int icount = 0;
        /*std::cout << line << std::endl;*/
        for (std::string subline;std::getline(subinput, subline,' ');icount++) {
            if (icount >= 8&&icount%2==0) {
                imageIds.push_back(std::stoi(subline));
            }
        }
        
        //iterate through and add to overlap matrix
        for (int i = 0; i < imageIds.size(); i++) {
            for (int j = i + 1; j < imageIds.size(); j++) {
                overlap_matrix(imageIds[i], imageIds[j]) += 1;
                overlap_matrix(imageIds[j], imageIds[i]) += 1;
            }
        }
    }
    return overlap_matrix;
}


Eigen::MatrixXd fileManager::read_imageId(std::string file_path, int n_images) {
    std::ifstream input(file_path, std::ifstream::in);
    std::unordered_map<int, int> imageId_to_idx;
    std::vector<std::vector<int>> points3D_id_to_2D;
    std::string line;
    Eigen::MatrixXd overlapMatrix = Eigen::MatrixXd::Zero(n_images, n_images);
    int count = 0;
    while (std::getline(input, line)) {
        if (count < 4) {
            count++;
            continue;
        }
        if (count % 2 == 0) {
            std::istringstream subinput(line);
            for (std::string subline; std::getline(subinput, subline, ' ');) {
                imageId_to_idx[std::stoi(subline)] = (count - 4)/int(2);
                break;
            }
        }
        else
        {
            int last = 0, next = 0;
            std::vector<std::string> strs;
            std::string subline;
            std::stringstream subinput(line);
            while (std::getline(subinput, subline, ' ')) {
                strs.push_back(subline);
            }
            /*while ((next = line.find(" ", last)) != string::npos)
            {
                strs.push_back(line.substr(last, next - last));
                last = next + 1;
            }*/
            std::vector<int> sub_points3D_id_to_2D;
            for (int i = 0; i < strs.size(); i++) {
                if ((i+1) % 3 == 0 && std::stoi(strs[i])!=-1) {
                    sub_points3D_id_to_2D.push_back(std::stoi(strs[i]));
                }
            }
            points3D_id_to_2D.push_back(sub_points3D_id_to_2D);
        }
        count++; 
    }
    //compute the overlap matrix
    for (int i = 0; i < n_images; i++) {
        for (int j = 0; j < n_images; j++) {
            std::vector<int> v_intersection;
            std::sort(points3D_id_to_2D[i].begin(), points3D_id_to_2D[i].end());
            std::sort(points3D_id_to_2D[j].begin(), points3D_id_to_2D[j].end());
            std::set_intersection(points3D_id_to_2D[i].begin(), points3D_id_to_2D[i].end(),
                points3D_id_to_2D[j].begin(), points3D_id_to_2D[j].end(),
                std::back_inserter(v_intersection));
            overlapMatrix(i, j) = (float)v_intersection.size() / points3D_id_to_2D[i].size();
            overlapMatrix(j, i) = (float)v_intersection.size() / points3D_id_to_2D[j].size();
        }
    }
    return overlapMatrix;
}
void fileManager::write_to_file(std::string name, std::vector<KeyPoint>& kpts, Mat& kCenters) {
    if (!fs::exists("Result")) {
        fs::create_directories("Result");
    }
    //keypoint write to file
    if (!kpts.empty()) {
        std::ofstream CSVOutput;
        int nKpts = kpts.size();
        CSVOutput.open(std::string("Result/" + name + "_statistics_" + std::to_string(nKpts) + "_" + helper::dateTime() + ".csv"), std::fstream::out | std::fstream::app);
        //input stream for headers
        CSVOutput << "Oritentation" << "," << "Oc`tave" << "," << "layer" << "," << "pt.x" << "," << "pt.y" << "," << "scale" << "\n";

        //write data to the file
        int numOkpt = kpts.size();
        int max_noctave = 0;
        for (int i = 0; i < numOkpt; i++) {
            auto kp = kpts[i];
            int noctave, nlayer;
            float vscale;
            helper::unpackOctave(kp, noctave, nlayer, vscale);
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

    String homo = parser.get<String>("homo");
    String tool = parser.get<String>("tool");

    if (homo == "true") {
        cmd.homo = true;
    }
    else if(homo !="false")
    {
        throw std::invalid_argument("-homo= " + homo + " is not supported");
    }
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
        else if (!fs::exists(test_path)) {
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
        std::cerr << "write graph: unsupported graph saving mode" << std::endl;
        throw std::invalid_argument("unsupported saving mode");
    }
}

void fileManager::read_user_set(fs::path params) {

    //read path
    json jsonlist;
    if (!fs::exists(params)) {
        cout << "Read Params: provided user setting .json doesn't exist!" << endl;
        throw std::invalid_argument("USER_SET_FILE_PATH_NOT_EXIST");
    }

    ifstream f(params.c_str());
    if (!f) {
        std::cerr << "Read Parameters: file read failed" << endl;
        throw std::runtime_error("USER_SET_FILE_READ_FAILED");
    }
    //check the status, throw exception for error
    f >> jsonlist;
    fileManager::parameters::userSetPath = jsonlist.value("userSetPath", fileManager::parameters::userSetPath);
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
    fileManager::parameters::siftPeakThres = jsonlist.value("siftPeakThres", 0.02/fileManager::parameters::noctaveLayer);
    fileManager::parameters::vladCenters = jsonlist.value("vladCenters", fileManager::parameters::vladCenters);
    fileManager::parameters::sampleSize = jsonlist.value("sampleSize", fileManager::parameters::sampleSize);
    fileManager::parameters::imgsetSize = jsonlist.value("imgsetSize", fileManager::parameters::imgsetSize);
    fileManager::parameters::PCliques = jsonlist.value("PCliques", fileManager::parameters::PCliques);
    fileManager::parameters::PCommonwords = jsonlist.value("PCommonwords", fileManager::parameters::PCommonwords);
    fileManager::parameters::maxNumOrient = jsonlist.value("maxNumOrient", fileManager::parameters::maxNumOrient);
    fileManager::parameters::maxNumFeatures = jsonlist.value("maxNumFeatures", fileManager::parameters::maxNumFeatures);
    fileManager::parameters::tfidfPath = jsonlist.value("tfidfPath", fileManager::parameters::tfidfPath);
}

/********class graphManager********/
fileManager::graphManager::graphManager(std::string root_path) {
    root_path_ = (fs::path(root_path)/"graphs").string();
    igraph_i_set_attribute_table(&igraph_cattribute_table);
    //scan the path
    std::vector<std::string> files_path;
    std::vector<std::string> allowed_extension = { "bin" };
    fileManager::read_files_in_path(root_path_, this->_graph_names, allowed_extension);
    for (auto& name : _graph_names) {
        name = fs::path(name).stem().string();
    }

}
bool fileManager::graphManager::Write(igraph_t mygraph) {

    if (!igraph_cattribute_has_attr(&mygraph, IGRAPH_ATTRIBUTE_VERTEX, "label")) {
        std::cerr << "\ngraphManager.writeGraph: missing \"label\" vertex attributes.\n";
        return false;
    }
    if (!igraph_cattribute_has_attr(&mygraph, IGRAPH_ATTRIBUTE_EDGE, "weight")) {
        std::cerr << "\ngraphManager.writeGraph: missing \"weight\" edge attributes.\n";
        return false;
    }
    if (GAN(&mygraph, "n_vertices") == 0) {
        std::cerr << "\ngraphManager.writeGraph: empty graph, stop writeing.\n";
        return false;
    }
    if (GAS(&mygraph, "name") == "") {
        std::cerr << "\ngraphManager.writeGraph: no graph name specified.\n";
        return false;
    }
    json outputBuffer;
    std::string graph_name_ = GAS(&mygraph, "name");
    outputBuffer["name"] = graph_name_;

    //write vertices
    outputBuffer["n_vertices"] = (int)GAN(&mygraph, "n_vertices");
    //write label
    std::unique_ptr<igraph_vector_t, void(*)(igraph_vector_t*)> labels(new igraph_vector_t(), &igraph_vector_destroy);
    igraph_vector_init(labels.get(), 0);
    VANV(&mygraph, "label", labels.get());
    std::vector<int> lab_vec;
    lab_vec.reserve(igraph_vector_size(labels.get()));
    for (int i = 0; i < igraph_vector_size(labels.get()); i++) {
        lab_vec.push_back(VECTOR(*labels)[i]);
    }
    if (!writeLabels(lab_vec,outputBuffer))
        return false;

    //write weighrs
    std::unique_ptr<igraph_vector_t, void(*)(igraph_vector_t*)> weights(new igraph_vector_t(), &igraph_vector_destroy);
    igraph_vector_init(weights.get(), 0);
    EANV(&mygraph, "weight", weights.get());
    std::vector<int> wei_vec;
    wei_vec.reserve(igraph_vector_size(weights.get()));
    for (int i = 0; i < igraph_vector_size(weights.get()); i++) {
        wei_vec.push_back(VECTOR(*weights)[i]);
    }
    if (!writeWeights(wei_vec,outputBuffer)) {
        return false;
    }

    //write Edges
    std::unique_ptr<igraph_vector_t, void(*)(igraph_vector_t*)> edges(new igraph_vector_t(), &igraph_vector_destroy);
    igraph_vector_init(edges.get(), 0);
    igraph_get_edgelist(&mygraph, edges.get(), false);
    std::vector<int> edge_vec;
    edge_vec.reserve(igraph_vector_size(edges.get()));
    for (int i = 0; i < igraph_vector_size(edges.get()); i++) {
        edge_vec.push_back(VECTOR(*edges)[i]);
    }
    if (!writeEdges(edge_vec,outputBuffer)) {
        return false;
    }

    auto jsonbin = json::to_msgpack(outputBuffer);
    
    //write to file
    std::ofstream fp;
    //check the existence of root_path_
    if (!fs::exists(fs::path(this->root_path_))) {
        fs::create_directories(this->root_path_);
    }

    std::ostringstream pcommon, pcliques;
    pcommon.precision(3), pcliques.precision(3);
    pcommon << std::fixed << parameters::PCommonwords;
    pcliques << std::fixed << parameters::PCliques;
    std::string full_graph_name = graph_name_ + "_" + std::to_string(parameters::maxNumFeatures) + "_" + pcommon.str() + "_" + pcliques.str();
    std::string file_path = (fs::path(this->root_path_) / (full_graph_name + ".bin")).string();
    fp.open(file_path, std::ios::out | std::ios::binary);
    fp.write((char*)jsonbin.data(), jsonbin.size());

    //add to the file_path
    if (std::find(_graph_names.begin(), _graph_names.end(), full_graph_name) == _graph_names.end()) {
        _graph_names.push_back(full_graph_name);
    }
    return true;
}
bool fileManager::graphManager::writeLabels(const std::vector<int>& labels, json& json_buffer) {
    if (json_buffer.contains("label"))
        json_buffer.erase("label");
    json_buffer["label"] = labels;
    return true;
}

bool fileManager::graphManager::writeEdges(const std::vector<int>& edges, json& json_buffer) {
    if (json_buffer.contains("edge"))
        json_buffer.erase("edge");
    json_buffer["edge"] = edges;
    return true;
}
bool fileManager::graphManager::writeWeights(const std::vector<int>& weights, json& json_buffer) {
    if (json_buffer.contains("weight"))
        json_buffer.erase("weight");
    json_buffer["weight"] = weights;
    return true;
}

//name should be the image name
bool fileManager::graphManager::Read(igraph_t* mygraph, std::string name) {
    //search on the folder
    json inputBuffer;

    //in current graphs folder can't find the corresponding graph with given name
    if (std::find(this->_graph_names.begin(), this->_graph_names.end(), name)==this->_graph_names.end()) {
        //corresponding graph Not found return false signal
        return false;
    }
    fs::path graph_path = fs::path(this->root_path_) / fs::path(name+ ".bin");

    std::ifstream fp(graph_path.string(), std::ios::binary);
    std::vector<uint8_t> gBson(
        (std::istreambuf_iterator<char>(fp)),
        std::istreambuf_iterator<char>());
    //read to buffer
    inputBuffer = json::from_msgpack(gBson);

    //check properties
    if (!inputBuffer.contains("name")) {
        std::cerr << "\ngraphManager.readGraph: corrupted graph, missing graph name\n";
        return false;
    }

    if (!inputBuffer.contains("n_vertices")) {
        std::cerr << "\ngraphManager.readGraph: corrupted graph, missing graph name\n";
        return false;
    }

    if (!inputBuffer.contains("label") || !inputBuffer.contains("weight") || !inputBuffer.contains("edge")) {
        std::cerr << "\ngraphManager.readGraph:corrupted graph, missing graph's components.\n";
        return false;
    }

    //write to graph
    //get edges
    auto edges = inputBuffer["edge"].get<std::vector<double>>();
    igraph_vector_t mygraph_edges;
    igraph_vector_view(&mygraph_edges, edges.data(), edges.size());

    //POSSIBLE isolated vertcies exists
    igraph_create(mygraph, &mygraph_edges, inputBuffer["n_vertices"].get<int>(), IGRAPH_UNDIRECTED);
    SETGAS(mygraph, "name", inputBuffer["name"].get<std::string>().c_str());
    SETGAN(mygraph, "n_vertices", inputBuffer["n_vertices"].get<int>());

    //set labels
    auto labels = inputBuffer["label"].get<std::vector<double>>();
    igraph_vector_t mygraph_labels;
    igraph_vector_view(&mygraph_labels, labels.data(), labels.size());
    SETVANV(mygraph, "label", &mygraph_labels);

    //set weights
    auto weights = inputBuffer["weight"].get<std::vector<double>>();
    igraph_vector_t mygraph_weights;
    igraph_vector_view(&mygraph_weights, weights.data(), weights.size());
    SETEANV(mygraph, "weight", &mygraph_weights);
    
    return true;
}

/*
    A test debug function
*/
void filecreate() {
    boost::filesystem::path curPath = boost::filesystem::current_path();
    boost::filesystem::path resultPath;
    for (auto it : boost::filesystem::recursive_directory_iterator(curPath)) {
        if (it.path().stem().string() == "vsImgProject") {
            resultPath = it.path();
            break;
        }
    }
    boost::filesystem::create_directory(resultPath.parent_path() / "Result");
}