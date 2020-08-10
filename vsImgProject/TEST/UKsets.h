#pragma once
#ifndef _UKSETS_H
#define _UKSETS_H

#include <string>
#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <StaticVRImg/extractor.h>
#include <StaticVRImg/cluster.h>
#include <StaticVRImg/fileManager.h>
#include <StaticVRImg/probModel.h>
#include <StaticVRImg/matcher.h>

namespace fs = std::filesystem;
using catPaths = std::vector<std::string>;
namespace UKB {
	class UKBench
	{
	public:
		UKBench() :categoryNum(0) {};
		UKBench(std::string& path, int catNum);
		~UKBench();
		int UKBdataRead(std::string path);
		void UKBcomputeTFIDF(probModel::tfidf& ukbTFIDF, matcher::kdTree& kdtreeMatcher);
		void UKdataExt(int begin, int end, std::vector<std::string>& imgs, std::vector<int>& indexs); //return begin~end opencv mat images
		void UKdataTrain(int n_catTrain);
		void UKdataSample(int size, std::vector<std::string>& imgs, std::vector<int>& indexes);

		std::vector<catPaths> imgPaths;
		std::vector<int> imgIndexs;
		int categoryNum;
		;
	};

	

	void UKFLANNTest(int argc, const char* argv[], int sampleSize, int imgsetSize);
	void UKVladTest(int argc, const char* argv[], int sampleSize, int imgsetSize);
	void UKtrain(int argc, const char* argv[], int numOfTrain);
	void UKtest(int argc, const char* argv[], int sampleSize, int imgsetSize);
	void UKResWriter(std::string name, std::vector<double> UKBScores);
}

#endif //!_UKSETS_H
