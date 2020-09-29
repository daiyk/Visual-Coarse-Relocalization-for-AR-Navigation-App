#pragma once
#ifndef _nbhdGraph_H
#define _nbhdGraph_H
#include <StaticVRImg/fileManager.h>
#include <StaticVRImg/matcher.h>
#include <StaticVRImg/kernel.h>
#include <colmap/base/database.h>
#include <colmap/util/bitmap.h>
namespace nbhd {
	class nbhdGraph {
	public:
		nbhdGraph();
		nbhdGraph(const fileManager::covisOptions& option);
		~nbhdGraph();
		std::vector<cv::Mat> Read(const std::vector<std::string> &paths);
		void preprocess();
		cv::Mat getVocab();
		void setVocab(std::string vocabPath);
		void setVocab(cv::Mat vocab);
		int NumDatabse() { if (database) { return database->NumImages(); } }
		int NumQuery() { return query_images.size(); }
		int NextIndex() { return next_index_; }
		int Next();
	private:
		int next_index_=-1;
		cv::Mat readVocab(std::string vocabPath);
		std::unique_ptr<matcher::kdTree> vocab;
		std::unique_ptr<kernel::covisMap> map;
		colmap::Database* database;
		std::vector<igraph_t> query_graphs;
		std::vector<cv::Mat> query_images;
		std::vector<std::string> query_image_names;
		std::string vocab_path_;

	};
}
#endif
