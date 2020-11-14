#pragma once
#ifndef _EVALUATION_H
#define _EVALUATION_H
#include <Eigen/Core>
#include <string>
#include "nbhdGraph.h"
#include "StaticVRImg/fileManager.h"
#include "StaticVRImg/vlad.h"
struct option_dataset {
	//path to a visual word dictionary
	std::string vocab_path = "";

	//path to the query folder
	std::string query_path = "";

	//path to the database
	std::string database_path = "";

	//path to the colmap database path
	std::string colmap_database_path = "";
	
};

class datasets {
public:
	datasets(option_dataset& option) { _option = option; }
	virtual ~datasets() = default;

	virtual bool preprocess() = 0;
	virtual bool Next() = 0;

	int numQuery() { return _num_query; }
	int numDatabase() { return _num_database; }

protected:
	int _num_query = 0;
	int _num_database = 0;
	option_dataset _option;
};

class grahamhall :public datasets {
public:
	grahamhall(option_dataset& option, fileManager::covisOptions& option_covis);
	virtual bool preprocess();
	virtual void vladTrain();
private:
	nbhd::nbhdGraph _nbhd_graph;
	fileManager::covisOptions _option_covis;
	int _next_index = -1;
};

/*** implementation ***/
grahamhall::grahamhall(option_dataset& option, fileManager::covisOptions& option_covis):datasets(option){
	_option_covis = option_covis;
}
bool grahamhall::preprocess() {
	//read image and store it
	std::vector<std::string> img_paths;
	fileManager::read_files_in_path(this->_option.database_path,img_paths);
	vlad::vlad
}

void grahamhall::vladTrain() {
	
}
#endif // !_EVAULATION_H