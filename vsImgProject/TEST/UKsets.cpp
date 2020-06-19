#include "UKsets.h"
#include "StaticVRImg/matcher.h"
#include "StaticVRImg/graph.h"
#include "StaticVRImg/kernel.h"
#include <random>
#include <omp.h>


UKB::UKBench::UKBench(std::string& path, int catNum)
{
	this->categoryNum = catNum;
	this->UKBdataRead(path);
}

UKB::UKBench::~UKBench()
{
}
int UKB::UKBench::UKBdataRead(std::string path) {
	//define the size of datasets
	if (imgPaths.size() != 0) {
		return 0;
	}
	this->imgPaths.resize(this->categoryNum);
	this->imgIndexs.reserve(this->categoryNum * 4);
	fs::path ukdata(path);
	if (!fs::is_directory(path)) {
		throw std::invalid_argument("please provide path to the training images directory!");
	}
	int count = 0, catInd = 0;
	for (const auto& it : fs::directory_iterator(ukdata)) {

		int ind = std::stoi(it.path().stem().string().substr(7, 5));
		int catInd = ind / int(4);
		this->imgPaths[catInd].push_back(it.path().string());

		//record indexes
		this->imgIndexs.push_back(catInd);
		//if the size surpass the defined size, then  break
		if (imgIndexs.size() > this->categoryNum) {
			break;
		}
		

	}
	std::cout << " --> successfully scan UKB images path" << " read "<<this->imgIndexs.size()<<" imgs"<<std::endl;
	return 1;
}
void UKB::UKBench::UKdataExt(int begin, int end, std::vector<std::string>& imgs, std::vector<int>& indexes) {
	if (begin > end) { throw std::invalid_argument("Begin should smaller than end for indexing!"); }
	auto first = this->imgPaths.begin()+begin;
	auto last = this->imgPaths.begin()+end;
	std::vector<catPaths> subs(first,last);
	indexes.reserve((end - begin) * 4);
	for (int i = 0; i < subs.size(); i++) {
		for (int j = 0; j < subs[i].size(); j++) {
			imgs.push_back(subs[i][j]);
			indexes.push_back(this->imgIndexs[begin+i*4+j]);
		}
	}
}

void UKB::UKBench::UKdataTrain(int n_catTrain) {
	//iterate through the trainingset and train the kcenter
	if (this->imgPaths.size() == 0) {
		std::cout << "UKB train: Error: please construct the dataset first" << std::endl;
		return;
	}
	if (n_catTrain > this->categoryNum) {
		std::cout << "UKB train: Error: indicate training categories is larger than the ukbsets" << std::endl;
		return;
	}

	std::cout << "!------ Start Training with UKB data ------!" << std::endl;

	int unit_train = 500;
	int n_iter = n_catTrain / unit_train, begin, end;
	cv::Mat allDescrips;
	for (int i = 0; i < n_iter + 1; i++) {

		begin = unit_train * i;
		end = unit_train * (i + 1);
		if (end > n_catTrain) { end = n_catTrain; }
		if (begin == end) { break; }
		std::vector<std::string> imgs;
		std::vector<int> indexs;
		this->UKdataExt(begin, end, imgs, indexs);

		//read and train descriptors, kpts
		cv::Mat descriptor;
		std::vector<cv::KeyPoint> kpts;
		try {
			extractor::vlimg_descips_compute(imgs, descriptor, kpts);
		}
		catch (std::invalid_argument& e) {
			std::cout << e.what() << std::endl;
			return;
		};

		std::cout << " --> finish feature detection for " << begin << " --- " << end << std::endl;
		//add to the total descriptors
		allDescrips.push_back(descriptor);
	}

	//kmeans clustering
	cv::Mat kCenters;
	cluster::vl_visual_word_compute(allDescrips, kCenters);

	//write to file
	std::vector<cv::KeyPoint> kpts;
	fileManager::write_to_file("UKB_vlfeat_" + std::to_string(n_catTrain), kpts, kCenters);

	std::cout << "!------ UKB training pogram ends here ------!" << std::endl;
}
void UKB::UKBench::UKdataSample(int size, std::vector<std::string>& imgs, std::vector<int>& indexes) {
	//sample the size number of imgs
	std::vector<int> pools(this->categoryNum);
	std::iota(pools.begin(), pools.end(), 0);

	//set seed
	std::vector<uint32_t> random_data(624);
	std::random_device source;
	std::generate(random_data.begin(), random_data.end(), std::ref(source));
	std::seed_seq seeds(random_data.begin(), random_data.end());
	std::mt19937 engine(seeds);
	indexes.clear();
	indexes.reserve(size);
	//sample
	std::sample(pools.begin(), pools.end(), std::back_inserter(indexes), size, engine);

	//for each category sample images
	imgs.clear();
	imgs.reserve(size);
	std::uniform_int_distribution<int> dist(0, 3);
	for (int i = 0; i < size; i++) {
		imgs.push_back(this->imgPaths[indexes[i]][dist(engine)]);
	}
}

//arg[1] is the path to UKB imagesets
//first argv should be the path to UKB image sets
void UKB::UKtrain(int argc, const char* argv[]) {
	std::string trainFolder = argv[1];
	UKBench ukb(trainFolder, 2550);
	ukb.UKdataTrain(2550);
}

//1th argv: path to the kcenter file
//2th argv: path to the UKB imagesets
void UKB::UKtest(int argc, const char* argv[]) {
	//sample test categories
	int sampleSize = 50; //find the best 4 for the sampled 50 images
	std::string testFolder = argv[2];
	double score_sum = 0.0;
	//read the database images
	UKBench ukb(testFolder, 2550);

	std::vector<std::vector<float>> scores(sampleSize,std::vector<float>(ukb.imgIndexs.size()));

	//read kcenters
	cv::FileStorage reader;
	reader.open(argv[1], cv::FileStorage::READ);
	if (!reader.isOpened()) { std::cout << "failed to open the kcenter file" << std::endl; return; }
	//read kcenters
	cv::Mat kCenters;
	reader["kcenters"] >> kCenters;
	reader.release();
	
	//start sampling 50 and get indexes
	std::vector<std::string> imgs;
	std::vector<int> indexes;
	ukb.UKdataSample(sampleSize, imgs, indexes);
	clock_t sTime = clock();

	//read queryimage and store them for furture comparison
	std::vector<igraph_t> query_graphs;

	//build graphs for query images and store them for comparison
	omp_set_num_threads(6);
	#pragma omp parallel
	{
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < sampleSize; i++) {
			std::vector<std::string> trainPath;
			trainPath.push_back(imgs[i]);
			cv::Mat descripts1;
			std::vector<cv::KeyPoint> kpts1;
			extractor::vlimg_descips_compute(trainPath, descripts1, kpts1);

			std::vector<cv::DMatch> matches = matcher::kdTree(kCenters, descripts1);
			igraph_t mygraph1;
			bool status = graph::build(matches, kpts1, mygraph1);
			if (!status) { std::cout << "graph build failed! check your function." << std::endl; }

			//store all query graphs and prepare for comparing with UKBdatasets
			query_graphs.push_back(mygraph1);
		}
	}

	//iterate the whole datasets and records the score

	for (int i = 0; i < ukb.imgPaths.size(); i++) {

		//iterate the graphs and compute the score
		for (int j = 0; j < ukb.imgPaths[i].size(); j++) {
			std::vector<std::string> trainPath;
			trainPath.push_back(ukb.imgPaths[i][j]);
			cv::Mat descripts2;
			std::vector<cv::KeyPoint> kpts2;
			extractor::vlimg_descips_compute(trainPath, descripts2, kpts2);

			//build graph and do comparing
			std::vector<cv::DMatch> matches = matcher::kdTree(kCenters, descripts2);
			igraph_t mygraph2;
			bool status = graph::build(matches, kpts2, mygraph2);
			if (!status) { std::cout << "graph build failed! check your function." << std::endl; }

			//do comparison for the query 50 imgs
			#pragma omp parallel
			{
				#pragma omp for schedule(dynamic)
				for (int k = 0; k < query_graphs.size(); k++) {
					//do comparison for the graphs
					kernel::robustKernel kernelObj(2, kCenters.rows);
					kernelObj.push_back(query_graphs[k]);
					kernelObj.push_back(mygraph2);
					auto resMat = kernelObj.robustKernelCom();
					//put score to the corresponding place
					scores[k][i * 4 + j] = MATRIX(resMat, 0, 1);
				}
			}
		}
	}
	
	//return the first 4 highest score categories for the query
	std::vector<int> indexes_local(ukb.imgIndexs.size());
	std::iota(indexes_local.begin(), indexes_local.end(), 0);
	for (int i = 0; i < sampleSize; i++) {
		std::vector<int> temp = indexes_local;

		std::sort(temp.begin(), temp.end(), [&](size_t left, size_t right) {return scores[i][left] > scores[i][right]; });

		//get the first 4 numbers
		double single_score = 0;
		for(int j = 0; j < 4; j++) {
			if (ukb.imgIndexs[temp[j]] == indexes[i]) {
				single_score += 1.0;
				
			}
		}
		score_sum += single_score;
		std::cout << "score for the sampled img " << i << " : " << single_score << std::endl;
	}
	std::cout << "total mean score is " << score_sum / 50.0 << std::endl;

}
