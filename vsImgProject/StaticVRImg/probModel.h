#pragma once
#ifndef _PROBMODEL_H
#define _PROBMODEL_H
#include <vector>
#include <opencv2/core.hpp>
namespace probModel {
	void processSampleLocation();
	//probably we only need idf instead of tf-idf, or only includes tf when we consider classification with category,
	//and each category with several images
	class tfidf {
	public:
		tfidf():dictSize(0),doc(0) { };
		tfidf(int size):doc(0) { dictSize = size; };
		void setNumDoc(int num) { doc = num;};
		void setNumDict(int dict) { dictSize = dict; };
		void init(){ weights = cv::Mat::zeros(cv::Size(dictSize, doc), CV_32F); }
		void addDoc(std::vector<cv::DMatch>& matches, int index); //add word statistics to the weights matrix
		void compute();
		void clear();
	private:
		int dictSize;
		int doc;
		cv::Mat weights;
	};
}

#endif // !_PROBMODEL_H

