#include "nbhdGraph.h"
#include <StaticVRImg/extractor.h>
#include <StaticVRImg/graph.h>
#include <StaticVRImg/helper.h>
#include <colmap/util/endian.h>
#include <opencv2/core/eigen.hpp>
#include <exe/colmap_util.h>
#include <fstream>
#include <map>
#include <chrono>
#include <omp.h>

void twoViewMatchesFilter(colmap::FeatureMatches& inlier_matches,int idx1_limit, int idx2_limit) {
	//filter out the matches that id exceeds the max_num_features
	//DEBUG: only works with max_num_features is set for both database and query graphs
	if (idx1_limit <= 0 || idx2_limit <= 0) {
		return;
	}
	colmap::FeatureMatches newMatches;
	newMatches.reserve(inlier_matches.size());
	for (int i = 0; i < inlier_matches.size(); i++) {
		if (inlier_matches[i].point2D_idx1-idx1_limit>0 && inlier_matches[i].point2D_idx2-idx2_limit>0) {
			newMatches.push_back(colmap::FeatureMatch(inlier_matches[i].point2D_idx1
				- idx1_limit, inlier_matches[i].point2D_idx2 - idx2_limit));
		}
	}
	newMatches.resize(newMatches.size());
	inlier_matches = newMatches;
}

std::vector<cv::DMatch> idsToMatches(const colmap::FeatureVisualIDs& ids, const colmap::image_t imageId) {
	std::vector<cv::DMatch> matches;
	colmap::FeatureVisualids featureIds = ids.ids;
	for (int i = 0; i < featureIds.rows(); i++) {
		cv::DMatch match(featureIds(i, 0), featureIds(i, 1), imageId, 0);
		matches.push_back(match);
	}
	return matches;
}

//cv::Mat bitmapToMat(colmap::Bitmap &bitmap) {
//	cv::Mat imgBitmap(bitmap.Height(), bitmap.Width(), CV_8U);
//	auto dataFloat = bitmap.ConvertToRowMajorArray();
//
//	//transform bitmap to opencv image
//	for (int i = 0; i < bitmap.Height(); i++) {
//		for (int j = 0; j < bitmap.Width(); j++) {
//			imgBitmap.at<uchar>(i, j) = dataFloat[j + i * bitmap.Width()];
//		}
//	}
//	return imgBitmap;
//}

nbhd::nbhdGraph::nbhdGraph(){ 

}
nbhd::nbhdGraph::nbhdGraph(const fileManager::covisOptions& options) {
	init(options);	
}


void nbhd::nbhdGraph::init(const fileManager::covisOptions& options) {
	//open the database
	database = new colmap::Database(options.database_path);

	//open the result writer
	this->write_stream_.open("evaluation_nbhd.csv", std::fstream::out);

	//scan the image list as query image
	this->query_image_names = options.image_list;

	//get the full image path
	std::vector<std::string> full_path;
	for (int i = 0; i < options.image_list.size(); i++) {
		full_path.push_back((boost::filesystem::path(options.image_path) / boost::filesystem::path(query_image_names[i])).string());
	}

	//we use each image as query unit
	this->query_images = this->Read(full_path);
	this->scores.resize(query_images.size()); //resize the score vector to the length of query images
	this->db_image_names.resize(query_images.size()); //resize the db_image_names to the query_size

	//read dictionary and build kdtree
	this->vocab_.Read(options.vocab_path);

	/*this->vocab = std::make_unique<matcher::kdTree>(readVocab(options.vocab_path));*/ // vlfeat kdtree
	this->vocab_path_ = options.vocab_path;

	//database transaction
	DatabaseTransaction transaction(database);

	//init covismap
	this->map = std::make_unique<kernel::covisMap>(this->vocab_.NumVisualWords(), database->NumImages() + 1);

	this->num_inliers_images_ = options.numImageToKeep;

	//max num of features allowed
	this->max_num_features_ = options.max_num_features;

	//construct the query graphs and store it
	this->preprocess(options);
}
nbhd::nbhdGraph::~nbhdGraph() {
	 database->Close();
	 delete graph_manager;
	 for (auto& i : query_graphs) {
		 igraph_destroy(&i);
	 }
}

//read image from the paths and store in the object
std::vector<cv::Mat> nbhd::nbhdGraph::Read(const std::vector<std::string> &paths) {
	std::vector<cv::Mat> query_images;
	if (paths.empty()) {
		std::cerr << "nbhd.read: Error: " << "path is empty!\n";
		return query_images;
	}
	
	for (int i = 0; i < paths.size(); i++) {
		//read all images
		colmap::Bitmap bitimg;
		if (!bitimg.Read(paths[i]))
		{
			std::cerr << "nbhd.read: Error: " << paths[i] << " read failed!\n";
			return query_images;
		}
		query_images.push_back(helper::bitmapToMat(bitimg));
		bitimg.Deallocate();
	}
	return query_images;
}

//vocabtree matching from query image features and build graphs on top of it 
void nbhd::nbhdGraph::preprocess(const fileManager::covisOptions& options) {
	//extract features from the query and build graphs
	this->query_graphs.clear();
	this->query_graphs.resize(NumQuery());

	//search on the database and check the existence
	if (use_vlfeat) {
		for (int i = 0; i < NumQuery(); i++) {
			if (this->database->ExistsImageWithName(this->query_image_names[i]))
			{
				
				int imageIds = this->database->ReadImageWithName(this->query_image_names[i]).ImageId();

				//read descriptor 
				auto visualIDs = this->database->ReadVisualIDs(imageIds).ids;
				
				std::vector < cv::DMatch> matches;
				matches.reserve(visualIDs.rows());
				for (int j = 0; j < visualIDs.rows(); j++) {
					matches.push_back(cv::DMatch(j, visualIDs(j, 1), -1));
				}
				std::vector<cv::KeyPoint> points;
				graph::buildFull(matches, points, query_graphs[i],this->query_image_names[i]);

			}
			else { //otherwise compute the descriptors for the new image
				auto qry_img = this->query_images[i];
				std::vector<cv::KeyPoint> points;
				cv::Mat descripts;
				extractor::vlimg_descips_compute_simple(qry_img, descripts, points);

				//matching with dictionary
				/*auto matches = this->vocab->colmapSearch(descripts);*/
				Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigenDescripts(descripts.rows, descripts.cols);
				for (int k = 0; k < descripts.rows; k++) {
					for (int j = 0; j < descripts.cols; j++) {
						eigenDescripts(k, j) = descripts.at<uint8_t>(k, j);
					}
				}
				int num_threads = -1;
				int num_checks = 256;
				int num_neighbors = 1;
				auto imageIds = this->vocab_.FindWordIds(eigenDescripts, num_neighbors, num_checks, num_threads);

				//transform back to opencv matches
				std::vector < cv::DMatch> matches;
				matches.reserve(descripts.rows);
				for (int j = 0; j < descripts.rows; j++) {
					//always the first one is the id because the eigen matrix is [rows,1] for single neighboor
					matches.push_back(cv::DMatch(j, imageIds(j, 0), -1));

				}
				graph::buildFull(matches, points, query_graphs[i],this->query_image_names[i]);
			}
		}
		////release images
		//for (auto i : this->query_images) {
		//	i.release();
		//}
	}
	else
	{
		if (options.exe_path == "") {
			std::cerr << "\nERROR: for colmap feature extraction, path to this porgram executable 'exe_path' must not be empty";
			return;
		}
		std::vector<colmap::FeatureDescriptors> query_descripts;
		std::vector<std::string> argStrs;
		int argv_count = 0;
		argStrs.push_back(options.exe_path);
		argStrs.push_back("--database_path");
		argStrs.push_back(options.database_path);
		argStrs.push_back("--image_path");

		argStrs.push_back(options.image_path);
		argStrs.push_back("--SiftExtraction.max_num_features");
		argStrs.push_back(std::to_string(this->max_num_features_));
		std::vector<char*> argChar;
		argChar.reserve(argStrs.size());
		for (int i = 0; i < argStrs.size(); i++) {
			argChar.push_back(const_cast<char*>(argStrs[i].c_str()));
		}
		colmap::RunSimpleFeatureExtractor(argChar.size(), argChar.data(), query_descripts);

		//find visual IDs
		int num_threads = -1;
		int num_checks = 256;
		int num_neighbors = 1;
		for (int i = 0; i < NumQuery(); i++) {
			auto imageIds = this->vocab_.FindWordIds(query_descripts[i], num_neighbors, num_checks, num_threads);
			
			//RunSimpleFeatureExtractor has defined the maxnum of features
			/*helper::ExtractTopFeatures(&(this->database->ReadKeypoints(i.first)), &base_graph_ids, fileManager::parameters::maxNumFeatures);
			auto matches = idsToMatches(base_graph_ids, i.first);*/
			std::vector < cv::DMatch> matches;
			std::vector<cv::KeyPoint> points;
			matches.reserve(query_descripts[i].rows());
			for (int j = 0; j < query_descripts[i].rows(); j++) {
				//always the first one is the id because the eigen matrix is [rows,1] for single neighboor
				matches.push_back(cv::DMatch(j, imageIds(j, 0), -1));
			}
			graph::buildFull(matches, points, query_graphs[i],this->query_image_names[i]);
		}
	}
	std::string graph_folder_path = fs::path(options.exe_path).parent_path().string();
	graph_manager = new fileManager::graphManager(graph_folder_path);

	//build invert_index_
	for (auto i : database->ReadAllImages()) {
		auto ids = database->ReadVisualIDs(i.ImageId());
		map->addEntry(i.ImageId(),ids);
	}
	//if preprocess success, set the next_index_
	this->next_index_ = 0;
	this->write_stream_ << "qry_img[ID]" << "," << "db_img[ID]" << "," << "score" << "\n";
}

auto nbhd::nbhdGraph::getVocab() {
	/*return this->vocab->getVocab();*/
	return this->vocab_.getVocab();
}

void nbhd::nbhdGraph::setVocab(std::string vocabPath) {
	//TODO change to colmapvisualindex class format
	this->vocab.reset();
	this->vocab = std::make_unique<matcher::kdTree>(readVocab(vocabPath));
}

void nbhd::nbhdGraph::setVocab(cv::Mat vocab) {
	//TODO change to colmapvisualindex class format
	this->vocab.reset();
	this->vocab = std::make_unique<matcher::kdTree>(vocab);
}
cv::Mat nbhd::nbhdGraph::readVocab(std::string vocabPath) {
	//TODO write overload for colmapvisualindex vocab
	std::ifstream file(vocabPath, std::ios::binary);
	CHECK(file.is_open()) << vocabPath;
	const uint64_t rows = colmap::ReadBinaryLittleEndian<uint64_t>(&file);
	const uint64_t cols = colmap::ReadBinaryLittleEndian<uint64_t>(&file);

	//read descriptors
	cv::Mat vocab(rows, cols, CV_8U);
	vocab.reserveBuffer(cols * rows); // reserve space for 
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			vocab.at<uint8_t>(i, j) = colmap::ReadBinaryLittleEndian<uint8_t>(&file);
		}
	}
	return vocab;
}

int nbhd::nbhdGraph::CompWithQueryArray() {
	DatabaseTransaction transaction(database);
	//process next_index_ query image and search locations in the database for the best image
	//build clique

	for (int next_index = 0; next_index < this->NumQuery(); next_index++) {
		std::unique_ptr<igraph_vector_t, void(*)(igraph_vector_t*)> labs(new igraph_vector_t(), &igraph_vector_destroy);
		igraph_vector_init(labs.get(), 0);
		VANV(&query_graphs[next_index], "label", labs.get());

		//from igraph labs vector to build FeatureVisualIDs
		colmap::FeatureVisualids visual_id;
		visual_id.resize(igraph_vector_size(labs.get()), Eigen::NoChange);
		for (int i = 0; i < igraph_vector_size(labs.get()); i++) {
			visual_id(i, 0) = i;
			visual_id(i, 1) = VECTOR(*labs)[i];
		}
		colmap::FeatureVisualIDs qry_id(visual_id, this->vocab_.NumVisualWords(), this->vocab_path_);
		//query the database and get candidate matching images
		std::vector<int> candidates;
		this->map->Query(qry_id, candidates);

		//better to isolate the graph extention test function
		//query database and do graph extention based on covisibility
		std::unordered_map<int, std::vector<int>> twoViewImages;
		this->computeExtList(candidates, twoViewImages);

		//init kernel for comparison
		kernel::recurRobustKel kernelobj(1, this->vocab_.NumVisualWords());
		kernelobj.push_back(query_graphs[next_index]);


		//*****do graph extention******//
		int gCount = 0;
		std::vector<int> twoViewImagesKeys;
		for (auto item : twoViewImages) {
			twoViewImagesKeys.push_back(item.first);
		}

		//resize the scores vector
		this->scores[next_index].resize(twoViewImagesKeys.size());
		this->db_image_names[next_index].resize(twoViewImagesKeys.size());
		int num_thread = 6;
		omp_set_num_threads(num_thread);
		#pragma omp parallel
		{
			#pragma omp for schedule(dynamic)
			for (auto t = 0; t < num_thread; t++) {
				int start = t * (twoViewImagesKeys.size() / num_thread);
				int end = (t + 1) * (twoViewImagesKeys.size() / num_thread);
				if (t == num_thread - 1) {
					end = twoViewImagesKeys.size();
				}
				std::vector<igraph_t> database_graphs;
				
				for (int i = start; i < end; i++) {
					//build graph for each candidate location
					//base graph is the key value graph
					//need to check the root_path first
					igraph_t database_graph;
					std::string db_img_name;
					#pragma omp critical(database)
					{
						db_img_name = fs::path(this->database->ReadImage(twoViewImagesKeys[i]).Name()).stem().string();
					}
					this->db_image_names[next_index][i] = db_img_name;
					std::ostringstream pcommon, pcliques;
					pcommon.precision(3), pcliques.precision(3);
					pcommon << std::fixed << fileManager::parameters::PCommonwords;
					pcliques << std::fixed << fileManager::parameters::PCliques;
					std::string db_graph_name = db_img_name + "_" + std::to_string(this->max_num_features_) + "_" + pcommon.str() + "_" + pcliques.str();
					if (!this->graph_manager->Read(&database_graph, db_graph_name)) {
						std::cout << "\nnbhdGraph.Next: cannot find corresponding graph in the folder, build the graph......";
						this->graphExtention(i, db_img_name, twoViewImagesKeys, twoViewImages, database_graph);
					}

					//do comparison with the base graph and stores the score
					
					database_graphs.push_back(database_graph);
					/*std::cout << "\nNumber " << gCount << " th image's graph extention passed";*/
					/*gCount++;
					if (gCount % 100 == 0) {
						this->write_stream_.flush();
					}
					igraph_destroy(&database_graph);*/
				}
				auto candidate_scores = kernelobj.robustKernelCompWithQueryArray(database_graphs);
				for (int j = start; j < end; j++) {
					this->scores[next_index][j] = candidate_scores[0][j];
				}
				for (auto& g : database_graphs) {
					igraph_destroy(&g);
				}
				
			}
		}
	}
	//write the result
	int dCount = 0;
	while (true) {
		bool status = false;
		for (int k = 0; k < this->scores.size(); k++) {
			if (dCount < this->scores[k].size()) {
				this->write_stream_ << this->query_image_names[k] << "," << this->db_image_names[k][dCount] << "," << this->scores[k][dCount] << ",";
				status = true;
			}
		}
		this->write_stream_ << "\n";
		dCount++;
		if (!status) {
			break;
		}
	}
	return -1;
}


int nbhd::nbhdGraph::Next() {
	if (next_index_ == -1) {
		return 0;
	}
	if (next_index_ == query_graphs.size()) {
		return 1;
	}
	//process next_index_ query image and search locations in the database for the best image
	//build clique
	
	std::unique_ptr<igraph_vector_t, void(*)(igraph_vector_t*)> labs(new igraph_vector_t(), &igraph_vector_destroy);
	igraph_vector_init(labs.get(), 0);
	VANV(&query_graphs[next_index_], "label", labs.get());

	//from igraph labs vector to build FeatureVisualIDs
	colmap::FeatureVisualids visual_id;
	visual_id.resize(igraph_vector_size(labs.get()),Eigen::NoChange);
	for (int i = 0; i < igraph_vector_size(labs.get()); i++) {
		visual_id(i, 0) = i;
		visual_id(i, 1) = VECTOR(*labs)[i];
	}
	colmap::FeatureVisualIDs qry_id(visual_id, this->vocab_.NumVisualWords(), this->vocab_path_);
	//query the database and get candidate matching images
	std::vector<int> candidates;
	this->map->Query(qry_id, candidates);

	//better to isolate the graph extention test function
	//query database and do graph extention based on covisibility
	std::unordered_map<int, std::vector<int>> twoViewImages;
	this->computeExtList(candidates, twoViewImages);

	//print out the inliers result
	/*for (auto i : twoViewImages) {
		std::cout << "\nthe "<< i.first << " th image result: ";
		for (auto j : i.second) {
			std::cout << j << " ";
		}
	}*/
	//init kernel for comparison
	kernel::robustKernel kernelobj(1, this->vocab_.NumVisualWords());
	/*kernel::recurRobustKel kernelobj(1, this->vocab_.NumVisualWords());*/
	kernelobj.push_back(query_graphs[next_index_]);

	
	//*****do graph extention******//
	int gCount = 0;
	std::vector<int> twoViewImagesKeys;
	for (auto item : twoViewImages) {
		twoViewImagesKeys.push_back(item.first); //twoViewImageKeys has stored all candidate image id
	}

	//resize the scores vector
	this->scores[next_index_].resize(twoViewImagesKeys.size());
	this->db_image_names[next_index_].resize(twoViewImagesKeys.size());
	omp_set_num_threads(6);
	#pragma omp parallel
	{
		#pragma omp for schedule(dynamic)
		for (auto i = 0; i < twoViewImagesKeys.size();i++) {
			//build graph for each candidate location
			//base graph is the key value graph
			//need to check the root_path first
			igraph_t database_graph;
			std::string db_img_name;
			#pragma omp critical(database)
			{
				db_img_name = fs::path(this->database->ReadImage(twoViewImagesKeys[i]).Name()).stem().string();
			}
			std::ostringstream pcommon, pcliques;
			pcommon.precision(3), pcliques.precision(3);
			pcommon << std::fixed << fileManager::parameters::PCommonwords;
			pcliques << std::fixed << fileManager::parameters::PCliques;
			std::string db_graph_name = db_img_name + "_" + std::to_string(this->max_num_features_) + "_" + pcommon.str() + "_" + pcliques.str();
			/*std::string db_graph_name = db_img_name + "Deg"+std::to_string(fileManager::parameters::maxNumDeg)+"_" + std::to_string(this->max_num_features_) + "_" + pcommon.str() + "_" + pcliques.str();*/

			if (!this->graph_manager->Read(&database_graph, db_graph_name)) {
				std::cout << "\nnbhdGraph.Next: cannot find corresponding graph in the folder, build the graph......";
				this->graphExtention(i, db_img_name,twoViewImagesKeys, twoViewImages, database_graph);
				/*this->graphExtentionWithRecurKernel(i, db_img_name, twoViewImagesKeys, twoViewImages, database_graph);*/

				/*std::string db_img_name;
				#pragma omp critical(database)
				{
					db_img_name = fs::path(this->database->ReadImage(twoViewImagesKeys[i]).Name()).stem().string();
				}
				std::ostringstream pcommon, pcliques;
				pcommon.precision(3), pcliques.precision(3);
				pcommon << std::fixed << fileManager::parameters::PCommonwords;
				pcliques << std::fixed << fileManager::parameters::PCliques;
				std::string db_graph_name = db_img_name + "_" + std::to_string(this->max_num_features_) + "_" + pcommon.str() + "_" + pcliques.str();
				igraph_t database_graph;
				if (!graph_manager->Read(&database_graph, db_graph_name)) {*/
				//std::cout << "\nnbhdGraph.Next: cannot find corresponding graph in the folder, build the graph......";
				////scan the folder and found 
				//colmap::FeatureVisualIDs base_graph_ids;
				//int total_num_feats_i;
				////from database read the corresponding images and 
				//#pragma omp critical
				//{
				//	base_graph_ids = this->database->ReadVisualIDs(twoViewImagesKeys[i]);
				//	total_num_feats_i = (base_graph_ids).ids.rows();
				//	helper::ExtractTopFeatures(&(this->database->ReadKeypoints(twoViewImagesKeys[i])), &base_graph_ids, this->max_num_features_);
				//}

				//auto matches = idsToMatches(base_graph_ids, twoViewImagesKeys[i]);
				//std::vector<cv::KeyPoint> pts;
				//graph::buildFull(matches, pts, database_graph,db_img_name);

				//if (GAN(&database_graph, "n_vertices") == 0) {
				//	std::cerr << "nbhdGraph.Next: error: base graph is empty!\n";
				//	break;
				//}

				////extend the graph
				//for (int j = 0; j < twoViewImages[twoViewImagesKeys[i]].size(); j++) {
				//	colmap::FeatureVisualIDs exd_graph_ids;
				//	colmap::FeatureMatches matchFeatId;
				//	int total_num_feats_j;
				//	std::string exd_graph_name;
				//	#pragma omp critical
				//	{
				//		exd_graph_name = fs::path(this->database->ReadImage(twoViewImages[twoViewImagesKeys[i]][j]).Name()).stem().string();
				//		exd_graph_ids = this->database->ReadVisualIDs(twoViewImages[twoViewImagesKeys[i]][j]);
				//		total_num_feats_j = exd_graph_ids.ids.rows();
				//		helper::ExtractTopFeatures(&(this->database->ReadKeypoints(twoViewImages[twoViewImagesKeys[i]][j])), &exd_graph_ids, this->max_num_features_);
				//		matchFeatId = database->ReadTwoViewGeometry(twoViewImagesKeys[i], twoViewImages[twoViewImagesKeys[i]][j]).inlier_matches;
				//	}

				//	auto exdmatches = idsToMatches(exd_graph_ids, twoViewImages[twoViewImagesKeys[i]][j]);

				//	igraph_t exd_graph;
				//	std::vector<cv::KeyPoint> pts;
				//	graph::buildFull(exdmatches, pts, exd_graph,exd_graph_name);

				//	if (GAN(&exd_graph, "n_vertices") == 0) {
				//		std::cerr << "nbhdGraph.Next: error: extend graph is empty!\n";
				//		break;
				//	}

				//	std::vector<cv::DMatch> twoViewMatches;

				//	//two view geometry includes all possible matches that some pairs id may exceed the maxNumFeat.
				//	//filter two view matches
				//	std::vector < cv::DMatch> newMatches;
				//	newMatches.reserve(matchFeatId.size());
				//	int idx1_limit = total_num_feats_i - this->max_num_features_;
				//	int idx2_limit = total_num_feats_j - this->max_num_features_;

				//	idx1_limit = idx1_limit > 0 ? idx1_limit : 0;
				//	idx2_limit = idx2_limit > 0 ? idx2_limit : 0;

				//	for (auto m : matchFeatId) {
				//		if (int(m.point2D_idx1) > idx1_limit && int(m.point2D_idx2) > idx2_limit) {
				//			newMatches.push_back(cv::DMatch(int(m.point2D_idx2) - idx2_limit, int(m.point2D_idx1)
				//				- idx1_limit, 0));
				//		}
				//	}

				//	if (!graph::extend1to2(database_graph, exd_graph, newMatches)) {
				//		std::cerr << "nbhdGraph: error: graph extention failed! with img id " << twoViewImagesKeys[i] << "\n";
				//		break;
				//	}
				//	igraph_destroy(&exd_graph);
				//}
				//write graph to the folder
			 /*	#pragma omp critical
				{
					graph_manager->Write(database_graph);
				}*/
				//}
			}

			//do comparison with the base graph and stores the score
			std::vector<igraph_t> database_graphs;
			database_graphs.push_back(database_graph);
			auto candidate_scores = kernelobj.robustKernelCompWithQueryArray(database_graphs);

			this->scores[next_index_][i]=candidate_scores[0][0];
			this->db_image_names[next_index_][i] = db_img_name;
			/*std::cout << "\nNumber " << gCount << " th image's graph extention passed";*/
			gCount++;
			if (gCount % 100 == 0) {
				this->write_stream_.flush();
			}
			igraph_destroy(&database_graph);
		}
	}

	//write the result
	for (int k = 0; k < scores[next_index_].size();k++) {
		this->write_stream_ << this->query_image_names[next_index_] << "," << this->db_image_names[next_index_][k]<< "," << this->scores[next_index_][k] << "\n";
	}
	
	next_index_++;
	return -1;
}

bool nbhd::nbhdGraph::graphExtention(int i, std::string db_img_name, std::vector<int> twoViewImagesKeys, std::unordered_map<int, std::vector<int>>& twoViewImages, igraph_t &database_graph) {
	//resize the scores vector

	//build graph for each candidate location
	//base graph is the key value graph
	//need to check the root_path first
		//scan the folder and found 
		colmap::FeatureVisualIDs base_graph_ids;
		int total_num_feats_i;
		colmap::FeatureKeypoints base_graph_kpts;
		//from database read the corresponding images and 
#pragma omp critical(database)
		{
			base_graph_ids = this->database->ReadVisualIDs(twoViewImagesKeys[i]);
			base_graph_kpts = this->database->ReadKeypoints(twoViewImagesKeys[i]);

		}
		total_num_feats_i = (base_graph_ids).ids.rows();
		helper::ExtractTopFeatures(&(base_graph_kpts), &base_graph_ids, this->max_num_features_);
		auto matches = idsToMatches(base_graph_ids, twoViewImagesKeys[i]);
		std::vector<cv::KeyPoint> pts;
		graph::buildFull(matches, pts, database_graph, db_img_name);

		if (GAN(&database_graph, "n_vertices") == 0) {
			std::cerr << "nbhdGraph.Next: error: base graph is empty!\n";
			return false;
		}

		//extend the graph
		for (int j = 0; j < twoViewImages[twoViewImagesKeys[i]].size(); j++) {
			colmap::FeatureVisualIDs exd_graph_ids;
			colmap::FeatureMatches matchFeatId;
			int total_num_feats_j;
			std::string exd_graph_name;
			colmap::FeatureKeypoints exd_graph_kpts;
#pragma omp critical(database)
			{
				exd_graph_name = fs::path(this->database->ReadImage(twoViewImages[twoViewImagesKeys[i]][j]).Name()).stem().string();
				exd_graph_ids = this->database->ReadVisualIDs(twoViewImages[twoViewImagesKeys[i]][j]);
				matchFeatId = database->ReadTwoViewGeometry(twoViewImagesKeys[i], twoViewImages[twoViewImagesKeys[i]][j]).inlier_matches;
				exd_graph_kpts = this->database->ReadKeypoints(twoViewImages[twoViewImagesKeys[i]][j]);
			}
			total_num_feats_j = exd_graph_ids.ids.rows();
			helper::ExtractTopFeatures(&(exd_graph_kpts), &exd_graph_ids, this->max_num_features_);

			auto exdmatches = idsToMatches(exd_graph_ids, twoViewImages[twoViewImagesKeys[i]][j]);

			igraph_t exd_graph;
			std::vector<cv::KeyPoint> pts;
			graph::buildFull(exdmatches, pts, exd_graph, exd_graph_name);

			if (GAN(&exd_graph, "n_vertices") == 0) {
				std::cerr << "nbhdGraph.Next: error: extend graph is empty!\n";
				return false;
			}
			std::vector<cv::DMatch> twoViewMatches;

			//two view geometry includes all possible matches that some pairs id may exceed the maxNumFeat.
			//filter two view matches
			std::vector < cv::DMatch> newMatches;
			newMatches.reserve(matchFeatId.size());
			int idx1_limit = total_num_feats_i - this->max_num_features_;
			int idx2_limit = total_num_feats_j - this->max_num_features_;

			idx1_limit = idx1_limit > 0 ? idx1_limit : 0;
			idx2_limit = idx2_limit > 0 ? idx2_limit : 0;

			for (auto m : matchFeatId) {
				if (int(m.point2D_idx1) > idx1_limit && int(m.point2D_idx2) > idx2_limit) {
					newMatches.push_back(cv::DMatch(int(m.point2D_idx2) - idx2_limit, int(m.point2D_idx1)
						- idx1_limit, 0));
				}
			}
			if (!graph::extend1to2(database_graph, exd_graph, newMatches)) {
				std::cerr << "nbhdGraph: error: graph extention failed! with img id " << twoViewImagesKeys[i] << "\n";
				return false;
			}
			igraph_destroy(&exd_graph);
		}
		//write graph to the folder
		this->graph_manager->Write(database_graph);
}

bool nbhd::nbhdGraph::graphExtentionWithRecurKernel(int i, std::string db_img_name, std::vector<int> twoViewImagesKeys, std::unordered_map<int, std::vector<int>>& twoViewImages, igraph_t& database_graph) {
	//resize the scores vector

	//build graph for each candidate location
	//base graph is the key value graph
	//need to check the root_path first
		//scan the folder and found 
	colmap::FeatureVisualIDs base_graph_ids;
	int total_num_feats_i;
	colmap::FeatureKeypoints base_graph_kpts;
	//from database read the corresponding images and 
#pragma omp critical(database)
	{
		base_graph_ids = this->database->ReadVisualIDs(twoViewImagesKeys[i]);
		base_graph_kpts = this->database->ReadKeypoints(twoViewImagesKeys[i]);

	}
	total_num_feats_i = (base_graph_ids).ids.rows();
	helper::ExtractTopFeatures(&(base_graph_kpts), &base_graph_ids, this->max_num_features_);
	auto matches = idsToMatches(base_graph_ids, twoViewImagesKeys[i]);
	std::vector<cv::KeyPoint> pts = helper::colmapToCvKpts(base_graph_kpts);
	graph::build(matches, pts, database_graph, db_img_name);

	if (GAN(&database_graph, "n_vertices") == 0) {
		std::cerr << "nbhdGraph.Next: error: base graph is empty!\n";
		return false;
	}

	//extend the graph
	for (int j = 0; j < twoViewImages[twoViewImagesKeys[i]].size(); j++) {
		colmap::FeatureVisualIDs exd_graph_ids;
		colmap::FeatureMatches matchFeatId;
		int total_num_feats_j;
		std::string exd_graph_name;
		colmap::FeatureKeypoints exd_graph_kpts;
#pragma omp critical(database)
		{
			exd_graph_name = fs::path(this->database->ReadImage(twoViewImages[twoViewImagesKeys[i]][j]).Name()).stem().string();
			exd_graph_ids = this->database->ReadVisualIDs(twoViewImages[twoViewImagesKeys[i]][j]);
			matchFeatId = database->ReadTwoViewGeometry(twoViewImagesKeys[i], twoViewImages[twoViewImagesKeys[i]][j]).inlier_matches;
			exd_graph_kpts = this->database->ReadKeypoints(twoViewImages[twoViewImagesKeys[i]][j]);
		}
		total_num_feats_j = exd_graph_ids.ids.rows();
		helper::ExtractTopFeatures(&(exd_graph_kpts), &exd_graph_ids, this->max_num_features_);

		auto exdmatches = idsToMatches(exd_graph_ids, twoViewImages[twoViewImagesKeys[i]][j]);

		igraph_t exd_graph;
		std::vector<cv::KeyPoint> pts = helper::colmapToCvKpts(exd_graph_kpts);;
		graph::build(exdmatches, pts, exd_graph, exd_graph_name);

		if (GAN(&exd_graph, "n_vertices") == 0) {
			std::cerr << "nbhdGraph.Next: error: extend graph is empty!\n";
			return false;
		}
		std::vector<cv::DMatch> twoViewMatches;

		//two view geometry includes all possible matches that some pairs id may exceed the maxNumFeat.
		//filter two view matches
		std::vector < cv::DMatch> newMatches;
		newMatches.reserve(matchFeatId.size());
		int idx1_limit = total_num_feats_i - this->max_num_features_;
		int idx2_limit = total_num_feats_j - this->max_num_features_;

		idx1_limit = idx1_limit > 0 ? idx1_limit : 0;
		idx2_limit = idx2_limit > 0 ? idx2_limit : 0;

		for (auto m : matchFeatId) {
			if (int(m.point2D_idx1) > idx1_limit && int(m.point2D_idx2) > idx2_limit) {
				newMatches.push_back(cv::DMatch(int(m.point2D_idx2) - idx2_limit, int(m.point2D_idx1)
					- idx1_limit, 0));
			}
		}
		if (!graph::extend1to2(database_graph, exd_graph, newMatches)) {
			std::cerr << "nbhdGraph: error: graph extention failed! with img id " << twoViewImagesKeys[i] << "\n";
			return false;
		}
		igraph_destroy(&exd_graph);
	}
	//write graph to the folder
	this->graph_manager->Write(database_graph);
}
void nbhd::nbhdGraph::computeExtList(std::vector<int> &candidates, std::unordered_map<int, std::vector<int>>& twoViewImages) {
	/*** second method ***/
	/*** num in the candidates are imageId not the index of image! ***/
	std::vector<std::pair<colmap::image_t, colmap::image_t>> image_pairs;
	std::vector<int> numInliers;
	database->ReadTwoViewGeometryNumInliers(&image_pairs, &numInliers);
	twoViewImages.clear();
	std::unordered_map<int, std::vector<int>> twoViewImagesNumInliners;
	for (auto i : candidates) {
		twoViewImages[i].reserve(this->num_inliers_images_);
		twoViewImagesNumInliners[i].reserve(this->num_inliers_images_);
	}
	//loop through the pairs and find the corresponding inlier image for merging need to forward and backward checking the numinliers
	for (int i = 0; i < image_pairs.size(); i++) {
		/*auto rel = std::find(candidates.begin(), candidates.end(), image_pairs[i].first);
		if (rel != candidates.end()) {
			twoViewInliers[image_pairs[i].first].insert({ numInliers[i], image_pairs[i].second });
		}*/
		if (twoViewImages.find(image_pairs[i].first) != twoViewImages.end()) {
			if (twoViewImages[image_pairs[i].first].size() < this->num_inliers_images_) {
				twoViewImages[image_pairs[i].first].push_back(image_pairs[i].second);
				twoViewImagesNumInliners[image_pairs[i].first].push_back(numInliers[i]);
			}
			else /*if(twoViewImagesNumInliners[image_pairs[i].first].back() < numInliers[i])*/ //else check by comparing through all stored elements
			{
				/*twoViewImagesNumInliners[image_pairs[i].first].pop_back();
				twoViewImages[image_pairs[i].first].pop_back();*/
				std::vector<int>::const_iterator it;
				for (it = twoViewImagesNumInliners[image_pairs[i].first].begin(); it != twoViewImagesNumInliners[image_pairs[i].first].end(); it++) {
					if (*it < numInliers[i]) {
						int offset = it - twoViewImagesNumInliners[image_pairs[i].first].begin();
						twoViewImagesNumInliners[image_pairs[i].first].insert(it, numInliers[i]); //insert the new elements and erase the old element
						twoViewImagesNumInliners[image_pairs[i].first].erase(twoViewImagesNumInliners[image_pairs[i].first].begin() + offset + 1);
						twoViewImages[image_pairs[i].first].insert(twoViewImages[image_pairs[i].first].begin() + offset, image_pairs[i].second);
						twoViewImages[image_pairs[i].first].erase(twoViewImages[image_pairs[i].first].begin() + offset + 1);
						break;
					}
				}

			}
		}
		if (twoViewImages.find(image_pairs[i].second) != twoViewImages.end()) {
			if (twoViewImages[image_pairs[i].second].size() < this->num_inliers_images_) {
				twoViewImages[image_pairs[i].second].push_back(image_pairs[i].first);
				twoViewImagesNumInliners[image_pairs[i].second].push_back(numInliers[i]);
			}
			else /*if (twoViewImagesNumInliners[image_pairs[i].second].back() < numInliers[i])*/
			{
				/*twoViewImagesNumInliners[image_pairs[i].second].pop_back();
				twoViewImages[image_pairs[i].second].pop_back();*/
				std::vector<int>::const_iterator it;
				for (it = twoViewImagesNumInliners[image_pairs[i].second].begin(); it != twoViewImagesNumInliners[image_pairs[i].second].end(); it++) {
					if (*it < numInliers[i]) {
						int offset = it - twoViewImagesNumInliners[image_pairs[i].second].begin();
						twoViewImagesNumInliners[image_pairs[i].second].insert(it, numInliers[i]);
						twoViewImagesNumInliners[image_pairs[i].second].erase(twoViewImagesNumInliners[image_pairs[i].second].begin() + offset + 1);
						twoViewImages[image_pairs[i].second].insert(twoViewImages[image_pairs[i].second].begin() + offset, image_pairs[i].first);
						twoViewImages[image_pairs[i].second].erase(twoViewImages[image_pairs[i].second].begin() + offset + 1);
						break;
					}
				}
			}
		}
	}
	twoViewImagesNumInliners.clear();
}



/***** we still want to keep the subset *****/
	////delete image set that is the subset of others
	//std::vector<int> deleted_set;
	//int nCount = 0;
	//for (auto it = twoViewImages.begin(); it != twoViewImages.end();it++) {
	//	for (int j = 1; j < twoViewImages.size() - nCount;j++) {
	//		auto it2 = std::next(it, j);
	//		
	//		//sort for equality comparison
	//		std::sort(it->second.begin(), it->second.end());
	//		std::sort(it2->second.begin(), it2->second.end());
	//		bool includes;
	//		if (it->second.size() > it2->second.size()) {
	//			includes = std::includes(it->second.begin(), it->second.end(), it2->second.begin(), it2->second.end());
	//			if (includes) {
	//				deleted_set.push_back(it2->first);
	//			}
	//		}
	//		else
	//		{
	//			includes = std::includes(it2->second.begin(), it2->second.end(), it->second.begin(), it->second.end());
	//			if (includes) {
	//				deleted_set.push_back(it->first);
	//			}
	//		}
	//		
	//	}
	//	nCount++;
	//}
	////delete set
	//for (auto& i : deleted_set) {
	//	twoViewImages.erase(i);
	//}
/*twoViewMatchesFilter(colmapMatches, total_num_feats_i - fileManager::parameters::maxNumFeatures,
total_num_feats_j - fileManager::parameters::maxNumFeatures);*/
/*igraph_vs_t vert1_nei;
igraph_vs_adj(&vert1_nei, 1, IGRAPH_ALL);
igraph_vit_t viti;
igraph_vit_create(&base_graph, vert1_nei, &viti);
std::vector<int> nei_vertices;
while (!IGRAPH_VIT_END(viti)) {
	nei_vertices.push_back((long int)IGRAPH_VIT_GET(viti));
	IGRAPH_VIT_NEXT(viti);
}*/


//look through each vector and keep only the best "NumInlierImages"
	//for (auto& it : twoViewInliers) {
	//	if (it.second.size() > this->num_inliers_images_-1) {
	//		auto iter = it.second.begin();
	//		std::advance(iter, num_inliers_images_-1);
	//		it.second.erase(iter, it.second.end());
	//	}
	//	std::cout << "\nthe candidate image set "<<it.first<<" are: ";
	//	for (auto it2 : it.second) {
	//		std::cout << it2.second << "  ";
	//		//insert all remain two view geometry images to the container for further process
	//		twoViewImages[it.first].push_back(it2.second);
	//	}
	//	std::cout << "\n";
	//}





//auto t1 = std::chrono::high_resolution_clock::now();
//std::unordered_map<int, std::multimap<int, int, std::greater<int>> > twoViewImagesMap;
//for (int i = 0; i < image_pairs.size(); i++) {
//	if (std::find(candidates.begin(), candidates.end(), image_pairs[i].first) != candidates.end()) {
//		if (twoViewImagesMap[image_pairs[i].first].size() < this->num_inliers_images_) {
//			twoViewImagesMap[image_pairs[i].first].insert({ numInliers[i],image_pairs[i].second });
//		}
//		else
//		{
//			if (twoViewImagesMap[image_pairs[i].first].rbegin()->first < numInliers[i]) //if key: numInliers smaller than the new inserted element
//			{
//				twoViewImagesMap[image_pairs[i].first].erase(std::next(twoViewImagesMap[image_pairs[i].first].rbegin()).base());
//				twoViewImagesMap[image_pairs[i].first].insert({ numInliers[i],image_pairs[i].second });
//			}
//		}
//	}
//	if (std::find(candidates.begin(), candidates.end(), image_pairs[i].second) != candidates.end()) {
//		if (twoViewImagesMap[image_pairs[i].second].size() < this->num_inliers_images_) {
//			twoViewImagesMap[image_pairs[i].second].insert({ numInliers[i],image_pairs[i].first });
//		}
//		else
//		{
//			if (twoViewImagesMap[image_pairs[i].second].rbegin()->first < numInliers[i]) //if key: numInliers smaller than the new inserted element
//			{
//				twoViewImagesMap[image_pairs[i].second].erase(std::next(twoViewImagesMap[image_pairs[i].second].rbegin()).base());
//				twoViewImagesMap[image_pairs[i].second].insert({ numInliers[i],image_pairs[i].first });
//			}
//		}
//	}
//}
////loop over the result
//std::unordered_map<int, std::vector<int>> twoViewImages;
//for (auto it : twoViewImagesMap) {
//	for (auto it2 : it.second) {
//		twoViewImages[it.first].push_back(it2.second);
//	}
//}
//twoViewImagesMap.clear();
//auto t2 = std::chrono::high_resolution_clock::now();
//std::cout << "\n 1th method time cost: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;


