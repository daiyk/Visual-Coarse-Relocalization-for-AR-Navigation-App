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
		void init(const fileManager::covisOptions& options);
		~nbhdGraph();
		std::vector<cv::Mat> Read(const std::vector<std::string> &paths);
		void preprocess(const fileManager::covisOptions& options);
		auto getVocab();
		void setVocab(std::string vocabPath);
		void setVocab(cv::Mat vocab);
		int NumDatabse() { if (database) { return database->NumImages(); } }
		int NumQuery() { return query_images.size(); }
		int NextIndex() { return next_index_; }
		auto getDatabase() { return this->database; }
		auto& getScores() { return scores; }
		int Next();
		int CompWithQueryArray();


	private:
		void computeExtList(std::vector<int>& candidates, std::unordered_map<int, std::vector<int>>& twoViewImages);
		bool graphExtention(int i, std::string db_img_name, std::vector<int> twoViewImagesKeys, std::unordered_map<int, std::vector<int>>& twoViewImages, igraph_t& database_graph);
		bool graphExtentionWithRecurKernel(int i, std::string db_img_name, std::vector<int> twoViewImagesKeys, std::unordered_map<int, std::vector<int>>& twoViewImages, igraph_t& database_graph);
		int next_index_=-1;
		int num_inliers_images_;
		int max_num_features_;
		bool use_vlfeat = false;
		fileManager::graphManager* graph_manager;
		cv::Mat readVocab(std::string vocabPath);
		colmap::Database* database = nullptr;
		matcher::colmapVisualIndex<> vocab_;
		std::unique_ptr<matcher::kdTree> vocab;
		std::unique_ptr<kernel::covisMap> map;
		std::vector<igraph_t> query_graphs;
		std::vector<cv::Mat> query_images;
		std::vector<std::string> query_image_names;
		std::vector<std::vector<std::string>> db_image_names;
		std::vector<std::vector<float>> scores;
		std::string vocab_path_;
		std::ofstream write_stream_;
		
	};
}
#endif
