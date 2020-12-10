#pragma once
#ifndef _EVALUATION_H
#define _EVALUATION_H
#include <Eigen/Core>
#include <string>
#include <bitset>
#include "StaticVRImg/fileManager.h"
#include "StaticVRImg/vlad.h"
#include "StaticVRImg/matcher.h"
#include "StaticVRImg/extractor.h"
#include "StaticVRImg/FLANN.h"
#include "StaticVRImg/helper.h"
#include "nbhdGraph.h"
#include "StaticVRImg/vlad.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <exe/colmap_util.h>

using params = fileManager::parameters;
struct option_feature {
	//number of scores
	int num_scores = 3;

	//exe path for colmap 
	std::string exe_path = "";

	//control parameter to control the methods for benchmark
	int level = 3;
};

struct option_dataset {
	//path to a visual word dictionary
	std::string vocab_path = "";

	//path to the query folder
	std::string query_path = "";

	//path to the database
	std::string database_path = "";

	//path to the colmap database path
	std::string colmap_database_path = "";

	//path to the vlad center file path
	std::string vlad_centers_path = "";

	//path to the vlad encoding of this datasets
	std::string vlad_encoding_path = "";
	
};

class datasets {
public:
	datasets(option_dataset& option) { 
		_option = option;
	}
	virtual ~datasets() = default;
	virtual bool preprocess() = 0;
	virtual int Next() = 0;
	virtual bool Read() = 0;
	int numQuery() { return _num_query; }
	int numDatabase() { return _num_database; }

protected:
	int _num_query = 0;
	int _num_database = 0;
	option_dataset _option;
	std::vector<std::string> _files_names;
};

class grahamhall :public datasets {
public:
	grahamhall(option_dataset& option, option_feature& option_feature_);
	~grahamhall();
	virtual bool preprocess();
	virtual bool Read();
	virtual int Next();
	virtual void vladTrain(); //vladTrain only requires the option_dataset.database_path and *.query_path,  and empty option_feature
	bool writeTofile();
private:
	int _next_index = -2;
	int _num_inlier_images;
	int _num_to_keep = 10;
	std::string _internal_control_value;
	nbhd::nbhdGraph* nbhd_kernel;
	vlad::vlad* vlad_matcher;
	std::vector<std::string> _query_paths;
	std::vector<std::string> _database_paths;
	std::string _exe_path; //path to the executable in case of colmap feature extraction
	std::vector<Eigen::MatrixXd> score_matrix;
	//std::vector<std::string> _query_image_names;
	//std::vector<igraph_t> _query_graphs;
	//matcher::colmapVisualIndex<> _vocab;
	
};

/*** implementation ***/
grahamhall::grahamhall(option_dataset& option_dataset_, option_feature& option_feature_):datasets(option_dataset_){
	//read files and store name and path
	this->_exe_path = option_feature_.exe_path;
	if (option_dataset_.database_path == "" || option_dataset_.query_path == "") {
		std::cerr << "\nERROR: dataset option is not complete";
		return;
	}
	_internal_control_value = std::bitset<3>(option_feature_.level).to_string();
}
grahamhall::~grahamhall() {
	delete nbhd_kernel;
	delete vlad_matcher;
}

bool grahamhall::Read() {
	if (_option.database_path.empty() || _option.query_path.empty()) {
		return false;
	}
	std::vector<std::string> database_imgs_path;
	
	colmap::Database database(this->_option.colmap_database_path);
	//read database instead! CAUTION: colmap database and database path need to be consistent!
	/*fileManager::read_files_in_path(_option.database_path, database_imgs_path);
	this->_num_database = database_imgs_path.size();
	this->_database_paths = database_imgs_path;*/
	this->_num_database = database.NumImages();
	this->_database_paths.reserve(this->_num_database);
	for (auto img : database.ReadAllImages()) {
		this->_database_paths.push_back(img.Name());
	}

	//read files for query
	std::vector<std::string> query_imgs_path;
	fileManager::read_files_in_path(_option.query_path, query_imgs_path);
	this->_num_query = query_imgs_path.size();
	this->_query_paths = query_imgs_path;

	//store query image name in the _file_names
	for (auto i : query_imgs_path) {
		this->_files_names.push_back(fs::path(i).filename().string());
	}
	
	_next_index = -1;

	return true;
}
bool grahamhall::preprocess() {
	//init nbhd graph kernel
	if (_next_index == -2) {
		this->Read();
		_next_index = -1;
	}
	fileManager::covisOptions option_covis;
	if (_option.colmap_database_path == "" || _option.vocab_path == "" || _option.vlad_centers_path == "" || _option.vlad_encoding_path == "") {
		std::cerr << "\nERROR: dataset option is not complete";
		return false;
	}
	if (_internal_control_value[2] - '0') {

		option_covis.database_path = this->_option.colmap_database_path;
		option_covis.exe_path = _exe_path;
		option_covis.vocab_path = this->_option.vocab_path;
		option_covis.image_path = this->_option.query_path;
		option_covis.max_num_features = fileManager::parameters::maxNumFeatures;
		for (int i = 0; i < this->numQuery(); i++) {
			option_covis.image_list.push_back(fs::path(this->_query_paths[i]).filename().string());
		}
		nbhd_kernel = new nbhd::nbhdGraph(option_covis);
	}
	if (_internal_control_value[0] - '0') {
		vlad_matcher = new vlad::vlad(_option.vlad_centers_path, _option.vlad_encoding_path);
	}
	
	//init score matrix
	//we use three methods
	score_matrix.resize(3);
	score_matrix[0]=Eigen::MatrixXd::Zero(_num_query, _num_to_keep); //vlad data
	score_matrix[1] = Eigen::MatrixXd::Zero(_num_database, _num_query); //FLANN
	score_matrix[2] = Eigen::MatrixXd::Zero(_num_database, _num_query); //nbhdgraph
	_next_index = 0;
}
int grahamhall::Next() {
	//iterate through the query image and compute the score
	//use vald to search for the nearest image in the database
	if (_next_index == -1) {
		std::cerr << "\nERROR: lack preprocess";
		return false;
	}
	//finish comparison for the three methods
	if (_next_index == 3) {
		return true;
	}
	//vlad search for the next query image
	
	//extract for vlad
	if (_next_index==0 && _internal_control_value[0] - '0') {
		std::vector<std::vector<int>> indices;
		std::vector<std::string> argStrs;
		std::vector<colmap::FeatureDescriptors> query_descripts;
		int argv_count = 0;
		argStrs.push_back(this->_exe_path);
		argStrs.push_back("--database_path");
		argStrs.push_back(this->_option.colmap_database_path);
		argStrs.push_back("--image_path");
		argStrs.push_back(this->_option.query_path);
		argStrs.push_back("--SiftExtraction.max_num_features");
		argStrs.push_back(std::to_string(fileManager::parameters::maxNumFeatures));
		std::vector<char*> argChar;
		argChar.reserve(argStrs.size());
		for (int i = 0; i < argStrs.size(); i++) {
			argChar.push_back(const_cast<char*>(argStrs[i].c_str()));
		}
		colmap::RunSimpleFeatureExtractor(argChar.size(), argChar.data(), query_descripts);
		for (int k = 0; k < this->_num_query; k++) {
			std::vector<int> ind;
			std::vector<double> scores;
			cv::Mat cv_query_descript;
			cv::eigen2cv(query_descripts[k], cv_query_descript);
			cv_query_descript.convertTo(cv_query_descript, CV_32F);
			vlad_matcher->searchWithDescripts(cv_query_descript, ind, scores, this->_num_to_keep);
			indices.push_back(ind);
			//assign the scores to the total scores
			for (int i = 0; i < this->_num_to_keep; i++) {
				this->score_matrix[0](k, i) = scores[i];
			}
		}
		//delete for reuse
		std::ofstream write_stream;
		write_stream.open("evaluation_vlad.csv", std::fstream::out);
		write_stream << "best_i_image[ID=" << this->_files_names[0] << "]" << "," << "scores[" << this->_files_names[0] << "]";
		for (int i = 1; i < this->_num_query; i++) {
			write_stream << ","<<"best_i_image[ID=" << this->_files_names[i] << "]"<<"," << "scores[" << this->_files_names[i] << "]";

		}
		write_stream <<"\n";
		for (int j = 0; j < this->_num_to_keep; j++)
		 {
			write_stream << this->_database_paths[indices[0][j]] << "," << this->score_matrix[0](0, j);
			for (int i = 1; i < this->_num_query; i++) {
				write_stream <<","<<this->_database_paths[indices[i][j]] << "," << this->score_matrix[0](i,j);
			}
			write_stream << "\n";
		}

		write_stream.flush();
		write_stream.close();
		/*vlad_matcher->~vlad();*/
	}
	
	//extract for flann
	if (_next_index == 1 && _internal_control_value[1] - '0') {
		//define flann process iterate through the database img
		//iterate the database img
		std::ofstream write_stream;
		std::vector<std::string> argStrs;
		std::vector<colmap::FeatureDescriptors> query_descripts;
		int argv_count = 0;
		argStrs.push_back(this->_exe_path);
		argStrs.push_back("--database_path");
		argStrs.push_back(this->_option.colmap_database_path);
		argStrs.push_back("--image_path");
		argStrs.push_back(this->_option.query_path);
		argStrs.push_back("--SiftExtraction.max_num_features");
		argStrs.push_back(std::to_string(fileManager::parameters::maxNumFeatures));
		std::vector<char*> argChar;
		argChar.reserve(argStrs.size());
		for (int i = 0; i < argStrs.size(); i++) {
			argChar.push_back(const_cast<char*>(argStrs[i].c_str()));
		}
		colmap::RunSimpleFeatureExtractor(argChar.size(), argChar.data(), query_descripts);

		colmap::Database database(this->_option.colmap_database_path);;
		
		write_stream.open("evaluation_flann.csv", std::fstream::out);
		write_stream << "db_img[ID]" << ",";
		for (int i = 0; i < this->numQuery(); i++) {
			write_stream << "score[" <<this->_files_names[i]<<"]"<< ",";
		}
		write_stream << "\n";
		int database_counter = 0; //count for the database comparison image index
		for (auto img : database.ReadAllImages()) {
			cv::Mat descrip_db_cv;
			auto colmap_descriptor = database.ReadDescriptors(img.ImageId());
			auto colmap_kpts = database.ReadKeypoints(img.ImageId());
			write_stream << img.Name();
			helper::ExtractTopDescriptors(&colmap_kpts, &colmap_descriptor, params::maxNumFeatures);
			cv::eigen2cv(colmap_descriptor, descrip_db_cv);
			for (int i = 0; i < query_descripts.size(); i++) {
				//compute the score
				cv::Mat descrip_qry_cv;
				cv::eigen2cv(query_descripts[i],descrip_qry_cv);

				std::vector<cv::DMatch> bestMatches;
				//ensure float datatype
				descrip_db_cv.convertTo(descrip_db_cv, CV_32F);
				descrip_qry_cv.convertTo(descrip_qry_cv, CV_32F);
				FLANN::FLANNMatch(descrip_qry_cv, descrip_db_cv, bestMatches);

				//score
				score_matrix[1](database_counter,i) = FLANN::FLANNScore(bestMatches);
				write_stream << "," << score_matrix[1](database_counter, i);

			}
			write_stream << "\n";
			database_counter++;
		}
		write_stream.close();
		
		
	}
	//colmap
	if (_next_index == 2 && _internal_control_value[2] - '0') {
		while (true) {
			int status = this->nbhd_kernel->Next();
			if (status == 0) {
				std::cerr << "nbhdGraphTest: error happens during graphObj operations" << std::endl;
				break;
			}
			if (status == 1) {
				break; //finish and break
			}
		}
		auto scores = this->nbhd_kernel->getScores();
		/*this->nbhd_kernel->CompWithQueryArray();*/
	}
	//stores the score
	_next_index++;
	return -1;
}

void grahamhall::vladTrain() {
	//read image and store it
	if (this->_option.colmap_database_path == "") {
		std::cout << "\nError: must provide colmap_database_path";
	}
	colmap::Database db(this->_option.colmap_database_path);
	//extract descripts for all database image;
	std::vector < cv::Mat > allDescripts;
	std::cout << "\nstart read descriptors from database......";
	for (auto img : db.ReadAllImages()) {
		cv::Mat cv_descript;
		auto colmap_descript = db.ReadDescriptors(img.ImageId());
		auto colmap_kpts = db.ReadKeypoints(img.ImageId());
		helper::ExtractTopDescriptors(&colmap_kpts, &colmap_descript, params::maxNumFeatures);
		cv::eigen2cv(colmap_descript, cv_descript);
		cv_descript.convertTo(cv_descript, CV_32F);
		allDescripts.push_back(cv_descript);
	}
	std::cout << "\nfinished reading, start training on vlad......";
	//transfer to opencv mat
	
	vlad::vlad vlad_trainer(allDescripts);
}

bool grahamhall::writeTofile() {
	//iterate through the score matrix
	std::ofstream write_stream;
	if (!this->score_matrix[0].isZero(0)) {
		
	}
	if (!this->score_matrix[1].isZero(0)) {
		write_stream.open("evaluation_vlad.csv", std::fstream::out | std::fstream::app);
		write_stream << "db_img[ID]" << ",";
		for (int i = 0; i < this->_num_to_keep; i++)
			write_stream << "top_" << i << "_score" << ",";
		write_stream << "\n";
	}
	
}
#endif // !_EVAULATION_H