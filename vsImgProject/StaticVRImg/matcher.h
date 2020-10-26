#pragma once

#ifndef _MATCHER_H
#define _MATCHER_H
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <FLANN/flann.hpp>
#include <feature/types.h>
#include <glog/logging.h>
//#include <retrieval/visual_index.h>
//#include <retrieval/inverted_file.h>
//#include <retrieval/inverted_index.h>
#include <util/endian.h>
extern "C" {
	#include "vl/kdtree.h"
}
using namespace colmap;

namespace matcher {
	struct matchOut {
		std::vector<cv::DMatch> matches;
		std::vector<cv::KeyPoint> source;
		std::vector<cv::KeyPoint> refer;
	};

	//from colmap visual_index, simplify to reserve wordId search only
	template <typename kDescType = uint8_t, int kDescDim = 128, int kEmbeddingDim = 64>
	class colmapVisualIndex {
	public:
		typedef Eigen::Matrix<kDescType, Eigen::Dynamic, kDescDim, Eigen::RowMajor>
			DescType;;
		typedef FeatureKeypoints GeomType;
		colmapVisualIndex() {};
		colmapVisualIndex(std::string vocabPath);
		~colmapVisualIndex();
        size_t NumVisualWords() const;
		void Read(const std::string& path);
		flann::Matrix<kDescType> getVocab() { return visual_words_; }
		Eigen::MatrixXi FindWordIds(const DescType& descriptors,
			const int num_neighbors, const int num_checks,
			const int num_threads) const;
	private:
		// The search structure on the quantized descriptor space.
		flann::AutotunedIndex<flann::L2<kDescType>> visual_word_index_;
		// The centroids of the visual words.
		flann::Matrix<kDescType> visual_words_;
	};

	class kdTree {
		public:
			kdTree(cv::Mat source);
			~kdTree();
			std::vector<cv::DMatch> search(cv::Mat& query);
			std::vector<cv::DMatch> colmapSearch(cv::Mat& query);
			cv::Mat getVocab() { return vocab; }
			size_t numWords() { return vocab_size_; }
			void setNN(int NN) { this->numOfNN = NN; }
			static matchOut kdTreeDemo(std::string& img1, std::string& img2, bool display = true);
		private:
			VlKDForest* tree=nullptr;
			size_t vocab_size_;
			int numOfNN;
			cv::Mat vocab;
	};

	std::vector<cv::DMatch> colmapFlannMatcher(const cv::Mat& query_descriptors, const cv::Mat& database_descriptors, int NNeighbors);

	std::vector<cv::DMatch> opencvFlannMatcher(cv::Mat& source, cv::Mat& query);
	void RANSC(cv::Mat& sourceDescrips, std::vector<cv::KeyPoint>& sourcekpts, cv::Mat& queryDescrips, std::vector<cv::KeyPoint>& querykpts, cv::Mat &mask, cv::Mat &homo);
}

//implementation
template <typename kDescType, int kDescDim, int kEmbeddingDim>
matcher::colmapVisualIndex<kDescType, kDescDim, kEmbeddingDim>::colmapVisualIndex(std::string vocabPath) { Read(vocabPath); }

template <typename kDescType, int kDescDim, int kEmbeddingDim>
matcher::colmapVisualIndex<kDescType, kDescDim, kEmbeddingDim>::~colmapVisualIndex() {
    if (visual_words_.ptr() != nullptr) {
        delete[] visual_words_.ptr();
    };
}


template <typename kDescType, int kDescDim, int kEmbeddingDim>
size_t matcher::colmapVisualIndex<kDescType, kDescDim, kEmbeddingDim>::NumVisualWords() const { return visual_words_.rows; }

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void matcher::colmapVisualIndex<kDescType, kDescDim, kEmbeddingDim>::Read(const std::string& path) {
    long int file_offset = 0;

    // Read the visual words.

    {
        if (visual_words_.ptr() != nullptr) {
            delete[] visual_words_.ptr();
        }

        std::ifstream file(path, std::ios::binary);
        CHECK(file.is_open()) << path;
        const uint64_t rows = ReadBinaryLittleEndian<uint64_t>(&file);
        const uint64_t cols = ReadBinaryLittleEndian<uint64_t>(&file);
        kDescType* visual_words_data = new kDescType[rows * cols];
        for (size_t i = 0; i < rows * cols; ++i) {
            visual_words_data[i] = ReadBinaryLittleEndian<kDescType>(&file);
        }
        visual_words_ = flann::Matrix<kDescType>(visual_words_data, rows, cols);
        file_offset = file.tellg();
    }

    // Read the visual words search index.

    visual_word_index_ =
        flann::AutotunedIndex<flann::L2<kDescType>>(visual_words_);

    {
        FILE* fin = fopen(path.c_str(), "rb");
        CHECK_NOTNULL(fin);
        fseek(fin, file_offset, SEEK_SET);
        visual_word_index_.loadIndex(fin);
        file_offset = ftell(fin);
        fclose(fin);
    }
}
template <typename kDescType, int kDescDim, int kEmbeddingDim>
Eigen::MatrixXi matcher::colmapVisualIndex<kDescType, kDescDim, kEmbeddingDim>::FindWordIds(const DescType& descriptors,
    const int num_neighbors, const int num_checks,
    const int num_threads) const {
    static_assert(DescType::IsRowMajor, "Descriptors must be row-major");

    CHECK_GT(descriptors.rows(), 0);
    CHECK_GT(num_neighbors, 0);

    Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        word_ids(descriptors.rows(), num_neighbors);
    word_ids.setConstant(-1);
    flann::Matrix<size_t> indices(word_ids.data(), descriptors.rows(),
        num_neighbors);

    Eigen::Matrix<typename flann::L2<kDescType>::ResultType, Eigen::Dynamic,
        Eigen::Dynamic, Eigen::RowMajor>
        distance_matrix(descriptors.rows(), num_neighbors);
    flann::Matrix<typename flann::L2<kDescType>::ResultType> distances(
        distance_matrix.data(), descriptors.rows(), num_neighbors);

    const flann::Matrix<kDescType> query(
        const_cast<kDescType*>(descriptors.data()), descriptors.rows(),
        descriptors.cols());
    flann::SearchParams search_params(num_checks);
    if (num_threads < 0) {
        search_params.cores = std::thread::hardware_concurrency();
    }
    else {
        search_params.cores = num_threads;
    }
    if (search_params.cores <= 0) {
        search_params.cores = 1;
    }

    visual_word_index_.knnSearch(query, indices, distances, num_neighbors,
        search_params);

    return word_ids.cast<int>();
}

#endif