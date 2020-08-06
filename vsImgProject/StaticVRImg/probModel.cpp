#include "probModel.h"
#include <iostream>
using namespace cv;
using namespace std;
void probModel::processSampleLocation() {

}
void probModel::tfidf::addDoc(vector<DMatch>& matches, int index) {
	//process and add word statistics to the matrix weight
	if (doc == 0 || dictSize == 0)
		std::cout << "ERROR: doc or dictsize is unset";
		return;
	for (size_t i = 0; i < matches.size; i++) {
		this->weights.at<float>(index, matches[i].trainIdx) += 1.0;
	}

}

void probModel::tfidf::compute() {
	//compute tf
	if (weights.empty() || dictSize == 0 || doc == 0) {
		std::cout << "ERROR: the object is uninitialized";
	}
	Mat term_freq = cv::Mat::zeros(this->weights.size(), CV_32F);
	Mat reduced_col;
	reduce(this->weights, reduced_col, 1, REDUCE_SUM);

	for (int i = 0; i < doc; i++) {
		for (int j = 0; j < dictSize; j++) {
			term_freq.at<float>(i, j) = this->weights.at<float>(i, j) / reduced_col.at<float>(i, 0);
		}
	}

	//compute inverse document freq (IDF)
	std::vector<float> inv_doc_freq(dictSize);
	for (int i = 0; i < dictSize; i++) {
		inv_doc_freq[i] = log((float)doc / (countNonZero(this->weights.col[i])+1));
	}
	
	//compute tf-idf weights
	for (int i = 0; i < doc; i++) {
		for (int j = 0; j < dictSize; j++) {
			this->weights.at<float>(i, j) = term_freq.at<float>(i, j) * inv_doc_freq[j];
		}
	}
}

void probModel::tfidf::clear() {
	weights.release();
	dictSize = 0;
	doc = 0;
}