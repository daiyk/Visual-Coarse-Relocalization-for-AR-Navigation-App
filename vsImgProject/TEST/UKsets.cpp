#include "UKsets.h"
#include "StaticVRImg/matcher.h"
#include "StaticVRImg/graph.h"
#include "StaticVRImg/kernel.h"
#include "StaticVRImg/vlad.h"
#include "StaticVRImg/FLANN.h"
#include "StaticVRImg/helper.h"
#include <random>
#include <omp.h>
#include <fstream>
#include <filesystem>

bool useCOvdet = false;
//1th arg: path to the ukbdata
void UKB::UKFLANNTest(int argc, const char* argv[], int sampleSize, int imgsetSize) {
	//read UKBench data
	//sample test categories
	//find the best 4 for the sampled images
	std::string testFolder = argv[1];
	double score_sum = 0.0;
	std::vector<double> single_scores(sampleSize, 0.0);
	//read the database images
	UKBench ukb(testFolder, imgsetSize);
	std::vector<std::vector<double>> scores(sampleSize, std::vector<double>(ukb.imgIndexs.size()));
	std::cout << " -->constructed test dataset with imgsets: " << ukb.imgIndexs.size() << std::endl;

	//start sampling and get indexes
	std::vector<std::string> imgs;
	std::vector<int> indexes;
	ukb.UKdataSample(sampleSize, imgs, indexes);
	std::cout << std::endl << " -->sampled " << sampleSize << " imgs from imgdatasets " << std::endl << " --> with category indexes: " << std::endl;
	for (auto i : imgs) {
		std::cout << i << std::endl;
	}

	std::vector<cv::Mat> allDescripts;
	//iterate through sampled imgs and store the descrips for comparison
	for (int i = 0; i < sampleSize; i++) {
		/*std::vector<std::string> trainPath;
		trainPath.push_back(imgs[i]);*/
		cv::Mat descripts1;
		std::vector<cv::KeyPoint> kpts1;

		if (!fs::exists(fs::path(imgs[i]))) {
			std::cout << "vlfeat sift feature detection: Warning: " << imgs[i] << " does not exist!" << std::endl;
			continue;
		}
		cv::Mat grayImg;
		cv::cvtColor(cv::imread(imgs[i]), grayImg, cv::COLOR_BGR2GRAY);

		try {
			if (useCOvdet) {
				extractor::covdetSIFT(grayImg, descripts1, kpts1);
			}
			else
				extractor::vlimg_descips_compute_simple(grayImg, descripts1, kpts1);
		}
		catch (std::invalid_argument& e) {
			std::cout << e.what() << std::endl;
			break;
		};
		allDescripts.push_back(descripts1);
	}

	//iterate through all ukb datasets and compute the score
	omp_set_num_threads(4);
#pragma omp parallel
	{
#pragma omp for schedule(dynamic)
		for (int i = 0; i < ukb.imgPaths.size(); i++) {
			//iterate the graphs and compute the score
			for (int j = 0; j < ukb.imgPaths[i].size(); j++) {
				if (!fs::exists(fs::path(ukb.imgPaths[i][j]))) {
					std::cout << "vlfeat sift feature detection: Warning: " << ukb.imgPaths[i][j] << " does not exist!" << std::endl;
					continue;
				}
				cv::Mat grayImg;
				cv::cvtColor(cv::imread(ukb.imgPaths[i][j]), grayImg, cv::COLOR_BGR2GRAY);

				cv::Mat descripts2;
				std::vector<cv::KeyPoint> kpts2;
				if ((4 * i + j) % 1000 == 0) {
					std::cout << " --> milestone " << 4 * i + j << " imgs" << std::endl;
				}
				try {
					if (useCOvdet)
						extractor::covdetSIFT(grayImg, descripts2, kpts2);
					else
						extractor::vlimg_descips_compute_simple(grayImg, descripts2, kpts2);
				}
				catch (std::invalid_argument& e) {
					std::cout << e.what() << std::endl;
					break;
				};
				//do comparison for all query images
				for (int k = 0; k < sampleSize; k++) {
					std::vector<cv::DMatch> matches;
					FLANN::FLANNMatch(descripts2, allDescripts[k], matches);
					scores[k][4 * i + j] = FLANN::FLANNScore(matches);
				}

			}
		}
	}
	std::vector<int> indexes_local(ukb.imgIndexs.size());
	std::iota(indexes_local.begin(), indexes_local.end(), 0);
	for (int i = 0; i < sampleSize; i++) {
		std::vector<int> temp(indexes_local);
		//cout score function
		/*std::cout << "before index rerank: "<<std::endl;
		for (int j = 0; j < scores[i].size(); j++) {
			std::cout << scores[i][j] << " ";
		}*/

		std::sort(temp.begin(), temp.end(), [&](size_t left, size_t right) {return scores[i][left] < scores[i][right]; });
		//report temp after rerank
		/*for (int j = 0; j < temp.size(); j++) {
			std::cout << temp[j] << " ";
		}*/

		/*std::cout << std::endl << std::endl;*/
		//get the first 4 numbers
		for (int j = 0; j < 4; j++) {
			if (ukb.imgIndexs[temp[j]] == indexes[i]) {
				single_scores[i] += 1.0;
			}
		}
		score_sum += single_scores[i];
	}
	std::cout << "total mean score is " << score_sum / sampleSize << std::endl;
	UKB::UKResWriter("UKFLANN_sample_" + std::to_string(sampleSize), single_scores);
}
//vlad test on ULB datasets
//1th arg: path to ukb full imgs folder path
//2th arg: path to the vlad center file
//3th arg" path to the encoding vector
void UKB::UKVladTest(int argc, const char* argv[], int sampleSize, int imgsetSize) {
	//read UKBench data
	//sample test categories
	//find the best 4 for the sampled 50 images
	std::string testFolder = argv[1];
	std::string kCenterPath = argv[2];
	std::string vladEncPath = argv[3];
	double score_sum = 0.0;
	std::vector<double> single_scores(sampleSize, 0.0);

	//read the database images
	UKBench ukb(testFolder, imgsetSize);
	std::cout << " -->constructed test dataset with imgsets: " << ukb.imgIndexs.size() << std::endl;

	//build vlad object
	vlad::vlad vladObj(kCenterPath, vladEncPath);

	//start sampling and get indexes
	std::vector<std::string> imgs;
	std::vector<int> indexes;
	ukb.UKdataSample(sampleSize, imgs, indexes);
	
	std::cout << std::endl << " -->sampled " << sampleSize << " imgs from imgdatasets " << std::endl << " --> with category indexes: " << std::endl;
	for (auto i : imgs) {
		std::cout << i << std::endl;
	}
	//read query images and do comparison
	for (int i = 0; i < sampleSize; i++) {
		/*std::vector<std::string> trainPath;
		trainPath.push_back(imgs[i]);*/
		cv::Mat descripts1;
		std::vector<cv::KeyPoint> kpts1;

		if (!fs::exists(fs::path(imgs[i]))) {
			std::cout << "vlfeat vlad: Warning: " << imgs[i] << " does not exist!" << std::endl;
			continue;
		}
		cv::Mat grayImg;
		cv::cvtColor(cv::imread(imgs[i]), grayImg, cv::COLOR_BGR2GRAY);
		std::vector<double> query_score;
		std::vector<int> query_ind;
		vladObj.search(grayImg, query_ind, query_score, 4);

		for (int j = 0; j < 4; j++) {
			int query_cat = query_ind[j] / int(4);
			if (query_cat == indexes[i]) {
				single_scores[i] += 1.0;
			}
		}
		score_sum += single_scores[i];
	}
	//output the final score
	std::cout << " The final score: " << score_sum / sampleSize << std::endl;
	//write down the final score
	UKB::UKResWriter("UKBVLAD_sample_" + std::to_string(sampleSize), single_scores);
}
//BBF-kmeans clustering
//arg[1] is the path to UKB imagesets
//first argv should be the path to UKB image sets
void UKB::UKtrain(int argc, const char* argv[], int numOfTrain) {
	std::string trainFolder = argv[1];
	UKBench ukb(trainFolder, numOfTrain);
	ukb.UKdataTrain(numOfTrain);
}

//1th argv: path to the UKB imagesets
//2th argv: path to the kcenter file
void UKB::UKtest(int argc, const char* argv[], int sampleSize, int imgsetSize) {
	//sample test categories
	//find the best 4 for the sampled 50 images
	bool useTFIDF = false; //whether include tfidf scores in the compuattion
	std::string testFolder = argv[1];
	double score_sum = 0.0;
	std::vector<double> single_scores(sampleSize, 0.0);
	//read the database images
	UKBench ukb(testFolder, imgsetSize);
	std::cout << " -->constructed test dataset with imgsets: " << ukb.imgIndexs.size() << std::endl;
	std::vector<std::vector<double>> scores(sampleSize, std::vector<double>(ukb.imgIndexs.size(),0.0));
	std::vector<std::vector<int>> raw_scores(sampleSize, std::vector<int>(ukb.imgIndexs.size(), 0));
	//read kcenters
	cv::FileStorage reader;
	reader.open(argv[2], cv::FileStorage::READ);
	if (!reader.isOpened()) { std::cout << "failed to open the kcenter file" << std::endl; return; }

	//read kcenters
	cv::Mat kCenters, tfidf;
	reader["kcenters"] >> kCenters;
	reader.release();

	//build kdtree
	matcher::kdTree kdtreeMatcher(kCenters);

	clock_t sTime = clock();
	kernel::robustKernel kernelCompObj(1, kdtreeMatcher.size());
	
	//read params file for read tfidf scores or compute tfidf score for the database
	if (!fileManager::parameters::tfidfPath.empty()) {
		reader.open(fileManager::parameters::tfidfPath, cv::FileStorage::READ);
		if (!reader.isOpened()) { std::cout << "failed to open the tfidf file" << std::endl; return; }

		//read kcenters
		reader["kcenters"] >> tfidf;
		reader.release();

		kernelCompObj.setTFIDF(tfidf);
	}
	else
	{
		probModel::tfidf ukbTFIDF;
		ukbTFIDF.setNumDict(kdtreeMatcher.size());
		ukbTFIDF.setNumDoc(imgsetSize);
		ukbTFIDF.init();
		ukb.UKBcomputeTFIDF(ukbTFIDF, kdtreeMatcher);
		ukbTFIDF.compute(); //compute tdidf score
		//write to file
		std::vector<cv::KeyPoint> kpts;
		fileManager::write_to_file("UKB_TFIDF_" + std::to_string(imgsetSize), kpts, ukbTFIDF.getWeights());
		//set tfidf scores
		kernelCompObj.setTFIDF(ukbTFIDF.getWeights());
}
	//start sampling and get indexes
	std::vector<std::string> imgs;
	std::vector<int> query_indexes;
	ukb.UKdataSample(sampleSize, imgs, query_indexes);
	std::cout << std::endl << " -->sampled " << sampleSize << " imgs from imgdatasets " << std::endl << " --> with category indexes: " << std::endl;
	for (auto i : imgs) {
		std::cout << i << std::endl;
	}
	std::cout << std::endl;

	//build graphs for query images and store them for comparison
	for (int i = 0; i < sampleSize; i++) {
		cv::Mat descripts1;
		std::vector<cv::KeyPoint> kpts1;

		if (!fs::exists(fs::path(imgs[i]))) {
			std::cout << "vlfeat sift feature detection: Warning: " << imgs[i] << " does not exist!" << std::endl;
			continue;
		}
		cv::Mat grayImg;
		grayImg=cv::imread(imgs[i],cv::IMREAD_GRAYSCALE);

		try {
			if (useCOvdet)
				extractor::covdetSIFT(grayImg, descripts1, kpts1);
			else
				extractor::vlimg_descips_compute_simple(grayImg, descripts1, kpts1);
		}
		catch (std::invalid_argument& e) {
			std::cout << e.what() << std::endl;
			break;
		};

		std::vector<cv::DMatch> matches = kdtreeMatcher.search(descripts1);
		igraph_t i_graph;
		bool status = graph::build(matches, kpts1, i_graph);
		if(useTFIDF)
			kernelCompObj.push_back(i_graph,query_indexes[i]);
		else
		{
			kernelCompObj.push_back(i_graph);
		}
		if (!status) { std::cout << "graph build failed! check your function." << std::endl; }

		//store all query graphs and prepare for comparing with UKBdatasets
	}
	std::cout << " -->finished query graphs building start iteration over database imgsets" << std::endl;
	auto& kernelgraphs = kernelCompObj.getGraphs();
	for (int i = 0; i < kernelgraphs.size(); i++) {
		std::cout << i << "th graph node number: " << igraph_vcount(&kernelgraphs[i]) << std::endl;
	}
	//iterate the whole datasets and records the score
	int unit_train = 50;
	int n_iter = imgsetSize / unit_train, begin, end;
	for (int i = 0; i < n_iter + 1; i++) {
		begin = unit_train * i;
		end = unit_train * (i + 1);
		if (end > imgsetSize) { end = imgsetSize; }
		if (begin == end) { break; }
		std::vector<std::string> imgs;
		std::vector<int> source_indexs;
		std::vector<igraph_t> databaseGraphs;
		std::cout << "the begin: " << begin << "\t the end: " << end << std::endl;
		ukb.UKdataExt(begin, end, imgs, source_indexs);

		databaseGraphs.resize(imgs.size());
#pragma omp parallel
		{
#pragma omp for schedule(dynamic)
			for (int j = 0; j < imgs.size(); j++) {
				if (!fs::exists(fs::path(imgs[j]))) {
					std::cout << "vlfeat sift feature detection: Warning: " << imgs[j] << " does not exist!" << std::endl;
					continue;
				}
				cv::Mat grayImg;
				grayImg = cv::imread(imgs[j], cv::IMREAD_GRAYSCALE);

				cv::Mat descripts2;
				std::vector<cv::KeyPoint> kpts2;
				try {
					if (useCOvdet)
						extractor::covdetSIFT(grayImg, descripts2, kpts2);
					else
						extractor::vlimg_descips_compute_simple(grayImg, descripts2, kpts2);
				}
				catch (std::invalid_argument& e) {
					std::cout << e.what() << std::endl;
					break;
				};
				//build graph and do comparing
				std::vector<cv::DMatch> matches = kdtreeMatcher.search(descripts2);
				bool status = graph::build(matches, kpts2, databaseGraphs[j]);
				if (!status) { std::cout << "graph build failed! check your function." << std::endl; }
			}
		}
		
		//compute scores for the query graphs, with tfidf weighting on
		auto scores_block = kernelCompObj.robustKernelCompWithQueryArray(databaseGraphs,&query_indexes,&source_indexs, useTFIDF);

		//assignment to the scores vector
		for(int m=0;m<sampleSize;m++)
			for (int n = 0; n < scores_block[m].size(); n++) {
				scores[m][n + begin * 4] = scores_block[m][n];
				raw_scores[m][n + begin * 4] = kernelCompObj.raw_scores[m][n];
			}
	}
	//for (int i = 0; i < ukb.imgPaths.size(); i++) {
	//	//iterate the graphs and compute the score
	//	for (int j = 0; j < ukb.imgPaths[i].size(); j++) {
	//		if (!fs::exists(fs::path(ukb.imgPaths[i][j]))) {
	//			std::cout << "vlfeat sift feature detection: Warning: " << ukb.imgPaths[i][j] << " does not exist!" << std::endl;
	//			continue;
	//		}
	//		cv::Mat grayImg;
	//		cv::cvtColor(cv::imread(ukb.imgPaths[i][j]), grayImg, cv::COLOR_BGR2GRAY);

	//		cv::Mat descripts2;
	//		std::vector<cv::KeyPoint> kpts2;
	//		if ((4 * i + j) % 1000 == 0) {
	//			std::cout << " --> milestone " << 4 * i + j << " imgs" << std::endl;
	//		}
	//		try {
	//			extractor::vlimg_descips_compute_simple(grayImg, descripts2, kpts2);
	//		}
	//		catch (std::invalid_argument& e) {
	//			std::cout << e.what() << std::endl;
	//			break;
	//		};
	//		//build graph and do comparing
	//		std::vector<cv::DMatch> matches = kdtreeMatcher.search(descripts2);
	//		igraph_t mygraph2;
	//		bool status = graph::build(matches, kpts2, mygraph2);
	//		if (!status) { std::cout << "graph build failed! check your function." << std::endl; }
	//		if (int(igraph_cattribute_GAN(&mygraph2, "vertices")) != 0)
	//		{
	//			/*fileManager::write_graph(mygraph2, fs::path(ukb.imgPaths[i][j]).stem().string(), "graphml");*/
	//		}
	//		//do comparison with whole query graphs
	//		for (int k = 0; k < query_graphs.size(); k++) {
	//			//do comparison for the graphs
	//			kernel::robustKernel kernelObj(1, kCenters.rows);

	//			//check if empty graph is tested
	//			if (int(igraph_cattribute_GAN(&query_graphs[k], "vertices")) == 0 || int(igraph_cattribute_GAN(&mygraph2, "vertices")) == 0) {
	//				scores[k][i * 4 + j] = 0.0;
	//				kernelObj.~robustKernel();
	//				continue;
	//			}

	//			kernelObj.push_back(query_graphs[k]);
	//			kernelObj.push_back(mygraph2);
	//			igraph_matrix_t resMat = kernelObj.robustKernelCom();
	//			//record score
	//			scores[k][i * 4 + j] = MATRIX(resMat, 0, 1);
	//			/*kernelObj.~robustKernel();*/
	//		}
	//		igraph_destroy(&mygraph2);
	//	}
	//}
	//
	////destory query_graphs
	//for (auto& i : query_graphs) {
	//	igraph_destroy(&i);
	//}

	//return the first 4 highest score categories for the query
	std::vector<int> indexes_local(ukb.imgIndexs.size());
	std::iota(indexes_local.begin(), indexes_local.end(), 0);
	for (int i = 0; i < sampleSize; i++) {
		std::vector<int> temp(indexes_local);
		//cout score function
		/*std::cout << "scores for every images: " << std::endl;
		for (int j = 0; j < scores[i].size(); j++) {
			std::cout << scores[i][j] << " ";
		}*/

		std::sort(temp.begin(), temp.end(), [&](size_t left, size_t right) {return scores[i][left] > scores[i][right]; });
		//report temp after rerank
		/*std::cout << "\n raw scores: \n";
		for (int j = 0; j < raw_scores[i].size(); j++) {
			std::cout << raw_scores[i][j] << " ";
		}*/
		//get the first 4 numbers
		for (int j = 0; j < 4; j++) {
			if (ukb.imgIndexs[temp[j]] == query_indexes[i]) {
				single_scores[i] += 1.0;
			}
		}
		std::cout << single_scores[i] << "  \n";
		score_sum += single_scores[i];
	}
	std::cout << "total mean score is " << score_sum / sampleSize << std::endl;
	UKB::UKResWriter("UKtest_sample_" + std::to_string(sampleSize), single_scores);
}


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
		if (it.is_directory()) {
			continue;
		}
		int ind = std::stoi(it.path().stem().string().substr(7, 5));
		int catInd = ind / int(4);
		if (catInd >= this->categoryNum) {
			continue;
		}
		this->imgPaths[catInd].push_back(it.path().string());
		//record indexes
		this->imgIndexs.push_back(catInd);
		//if the size surpass the defined size, then  break
	}
	std::cout << " --> successfully scan UKB images path" << " read "<<this->imgIndexs.size()<<" imgs"<<std::endl;
	return 1;
}
void UKB::UKBench::UKdataExt(int begin, int end, std::vector<std::string>& imgs, std::vector<int>& indexes) {
	if (begin > end) { throw std::invalid_argument("Begin should smaller than end for indexing!"); }
	auto first = this->imgPaths.begin() + begin;
	auto last = this->imgPaths.begin() + end;
	std::vector<catPaths> subs(first,last);
	indexes.reserve((end - begin) * 4);
	for (int i = 0; i < subs.size(); i++) {
		for (int j = 0; j < subs[i].size(); j++) {
			imgs.push_back(subs[i][j]);
			indexes.push_back(this->imgIndexs[(begin+i)*4+j]);
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

		std::cout << "the begin: " << begin << "\t the end: " << end << std::endl;
		this->UKdataExt(begin, end, imgs, indexs);

		//read and train descriptors, kpts
		cv::Mat descriptor;
		std::vector<cv::KeyPoint> kpts;

		for (int j = 0; j < imgs.size(); j++) {
			cv::Mat locDescrip;
			try {
				cv::Mat grayImg = cv::imread(imgs[j], cv::IMREAD_GRAYSCALE);
				if (useCOvdet)
					extractor::covdetSIFT(grayImg, locDescrip, kpts);
				else
					extractor::vlimg_descips_compute_simple(grayImg, locDescrip, kpts);
				descriptor.push_back(locDescrip);
			}
			catch (std::invalid_argument& e) {
				std::cout << e.what() << std::endl;
				return;
			};
		}
	
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
	if (size > this->categoryNum) {
		throw std::invalid_argument("size exceeds the image databse size");
	}

	std::vector<int> pools(this->categoryNum);
	std::iota(pools.begin(), pools.end(), 0);

	//set seed
	/*std::vector<uint32_t> random_data(624);
	std::random_device source;
	std::generate(random_data.begin(), random_data.end(), std::ref(source));
	std::seed_seq seeds(random_data.begin(), random_data.end());
	std::mt19937 engine(seeds);*/
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 engine(rd()); //Standard mersenne_twister_engine seeded with rd()
	indexes.clear();
	indexes.reserve(size);
	//sample
	std::sample(pools.begin(), pools.end(), std::back_inserter(indexes), size, engine);

	//for each category sample images
	imgs.clear();
	imgs.reserve(size);

	std::uniform_int_distribution<int> dist(0, 3);
	for (int i = 0; i < size; i++) {
		int ind1 = indexes[i], ind2 = dist(engine);
		imgs.push_back(this->imgPaths[ind1][ind2]);
	}
}

void UKB::UKResWriter(std::string name, std::vector<double> UKBScores) {
	std::ofstream CSVOutput;
	if (!fs::exists("Result")) {
		fs::create_directories("Result");
	}
	if (!UKBScores.empty()) {
		std::ofstream CSVOutput;
		int nScores = UKBScores.size();
		CSVOutput.open(std::string("Result/" + name + "_UKBScores_" + helper::dateTime() + ".csv"), std::fstream::out | std::fstream::app);
		//input stream for headers
		CSVOutput << name << "\n";

		//write scores to file
		for (int i = 0; i < nScores; i++) {
			CSVOutput << UKBScores[i] << "\n";
		}
	}
	CSVOutput.close();
	std::cout << " --> Finished writing UKBScore " << name << " :" << UKBScores.size() << std::endl;
}

void UKB::UKBench::UKBcomputeTFIDF(probModel::tfidf& ukbTFIDF, matcher::kdTree &kdtreeMatcher) {
	if (this->imgPaths.empty() || this->imgIndexs.empty()) {
		throw std::invalid_argument("ERROR: data is empty, read image data before preprocess!\n");
	}
	if (ukbTFIDF.empty()) { std::cout << "arg error: input tfidf object is uninitialized.\n"; }


	//iterate through the imgs which create the tfidf
	for (int i = 0; i < this->imgPaths.size(); i++) {
		for (int j = 0; j < this->imgPaths[i].size(); j++) {
			if (!fs::exists(fs::path(this->imgPaths[i][j]))) {
				std::cout << "vlfeat sift feature detection: Warning: " << this->imgPaths[i][j] << " does not exist!" << std::endl;
				continue;
			}
			cv::Mat grayImg;
			grayImg = cv::imread(this->imgPaths[i][j], cv::IMREAD_GRAYSCALE);

			cv::Mat descripts;
			std::vector<cv::KeyPoint> kpts;
			try {
				if (useCOvdet)
					extractor::covdetSIFT(grayImg, descripts, kpts);
				else
					extractor::vlimg_descips_compute_simple(grayImg, descripts, kpts);
			}
			catch (std::invalid_argument& e) {
				std::cout << e.what() << std::endl;
				break;
			};
			auto matches = kdtreeMatcher.search(descripts);
			ukbTFIDF.addDoc(matches, i);
		}	
	}
}


