#include "nbhdGraph.h"
#include <StaticVRImg/extractor.h>
#include <StaticVRImg/graph.h>
#include <StaticVRImg/helper.h>
#include <colmap/util/endian.h>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <map>

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

cv::Mat bitmapToMat(colmap::Bitmap &bitmap) {
	cv::Mat imgBitmap(bitmap.Height(), bitmap.Width(), CV_8U);
	auto dataFloat = bitmap.ConvertToRowMajorArray();

	//transform bitmap to opencv image
	for (int i = 0; i < bitmap.Height(); i++) {
		for (int j = 0; j < bitmap.Width(); j++) {
			imgBitmap.at<uchar>(i, j) = dataFloat[j + i * bitmap.Width()];
		}
	}

	return imgBitmap;
}

nbhd::nbhdGraph::nbhdGraph(){ 

}
nbhd::nbhdGraph::nbhdGraph(const fileManager::covisOptions& options) {
	//open the database
	database = new colmap::Database(options.database_path);

	//open the result writer
	this->write_stream_.open("nbhdTestResult.csv", std::fstream::out | std::fstream::app);

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

	//read dictionary and build kdtree
	this->vocab_ .Read(options.vocab_path);

	/*this->vocab = std::make_unique<matcher::kdTree>(readVocab(options.vocab_path));*/ // vlfeat kdtree
	this->vocab_path_ = options.vocab_path;

	//init covismap
	this->map = std::make_unique<kernel::covisMap>(this->vocab_.NumVisualWords(),database->NumImages()+1);

	this->num_inliers_images_ = options.numImageToKeep;

	//construct the query graphs and store it
	this->preprocess();
		
}
nbhd::nbhdGraph::~nbhdGraph() {
	 database->Close();
	 for (auto& i : query_graphs) {
		 igraph_destroy(&i);
	 }
	 this->write_stream_.close();
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
		query_images.push_back(bitmapToMat(bitimg));
		bitimg.Deallocate();
	}
	return query_images;
}

//vocabtree matching from query image features and build graphs on top of it 
void nbhd::nbhdGraph::preprocess() {
	//extract features from the query and build graphs
	this->query_graphs.clear();
	this->query_graphs.resize(NumQuery());

	//search on the database and check the existence
	for (int i = 0; i < NumQuery(); i++) {
		if(false)/*if (this->database->ExistsImageWithName(this->query_image_names[i]))*/
		{
			int imageIds = this->database->ReadImageWithName(this->query_image_names[i]).ImageId();

			//read descriptor 
			auto visualIDs = this->database->ReadVisualIDs(imageIds).ids;
			std::vector < cv::DMatch> matches;
			matches.reserve(visualIDs.rows());
			for (int i = 0; i < visualIDs.rows(); i++) {
				matches.push_back(cv::DMatch(i, visualIDs(i, 1), -1));
			}
			std::vector<cv::KeyPoint> points;
			graph::buildFull(matches, points, query_graphs[i]);

		}

		else { //otherwise compute the descriptors for the new image
			auto qry_img = this->query_images[i];
			std::vector<cv::KeyPoint> points;
			cv::Mat descripts;
			extractor::vlimg_descips_compute_simple(qry_img, descripts, points);

			//matching with dictionary
			/*auto matches = this->vocab->colmapSearch(descripts);*/
			Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigenDescripts(descripts.rows, descripts.cols);
			for (int i = 0; i < descripts.rows; i++) {
				for (int j = 0; j < descripts.cols; j++) {
					eigenDescripts(i, j) = descripts.at<uint8_t>(i, j);
				}
			}
			int num_threads = -1;
			int num_checks = 256;
			int num_neighbors = 1;
			auto imageIds = this->vocab_.FindWordIds(eigenDescripts, num_neighbors, num_checks, num_threads);

			//transform back to opencv matches
			std::vector < cv::DMatch> matches;
			matches.reserve(descripts.rows);
			for (int i = 0; i < descripts.rows; i++) {
				//always the first one is the id because the eigen matrix is [rows,1] for single neighboor
				matches.push_back(cv::DMatch(i, imageIds(i, 0), -1));

			}
			graph::buildFull(matches, points, query_graphs[i]);
		}
	}
	//release images
	for (auto i : this->query_images) {
		i.release();
	}

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

int nbhd::nbhdGraph::Next() {
	if (next_index_ == -1) {
		return 0;
	}
	if (next_index_ == query_images.size()) {
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

	//query database and do graph extention based on covisibility
	std::vector<std::pair<colmap::image_t, colmap::image_t>> image_pairs;
	std::vector<int> numInliers;
	database->ReadTwoViewGeometryNumInliers(&image_pairs, &numInliers);
	
	//container for the comparison of two view verified image id inliers
	/*std::unordered_map<int, std::multimap<int,int,std::greater<int>>> twoViewInliers;*/
	std::map<int, std::vector<int>> twoViewImages;
	
	for (auto i : candidates) {
		twoViewImages[i].reserve(this->num_inliers_images_);
	}
	//loop through the pairs and find the corresponding inlier image for merging
	for (int i = 0; i < image_pairs.size();i++) {
		/*auto rel = std::find(candidates.begin(), candidates.end(), image_pairs[i].first);
		if (rel != candidates.end()) {
			twoViewInliers[image_pairs[i].first].insert({ numInliers[i], image_pairs[i].second });
		}*/
		if (twoViewImages.find(image_pairs[i].first) != twoViewImages.end()) {
			if (twoViewImages[image_pairs[i].first].size() < this->num_inliers_images_) {
				/*std::cout << image_pairs[i].first << "th image numInliers: " << image_pairs[i].second << " with " << numInliers[i] << std::endl;*/
				twoViewImages[image_pairs[i].first].push_back(image_pairs[i].second);
			}
		}
	}

	/*twoViewInliers.clear();*/
	//init kernel for comparison
	kernel::robustKernel kernelobj(1, this->vocab_.NumVisualWords());
	kernelobj.push_back(query_graphs[next_index_]);

	
	//*****do graph extention******//
	int gCount = 0;
	for (auto& i : twoViewImages) {
		//build graph for each candidate location
		//base graph is the key value graph
		colmap::FeatureVisualIDs base_graph_ids = this->database->ReadVisualIDs(i.first);
		int total_num_feats_i = (base_graph_ids).ids.rows();

		helper::ExtractTopFeatures(&(this->database->ReadKeypoints(i.first)),&base_graph_ids,fileManager::parameters::maxNumFeatures);
		auto matches = idsToMatches(base_graph_ids, i.first);
		
		igraph_t base_graph;
		std::vector<cv::KeyPoint> pts;
		graph::buildFull(matches, pts, base_graph);
		
		if (GAN(&base_graph, "n_vertices") == 0) {
			std::cerr << "nbhdGraph.Next: error: base graph is empty!\n" ;
			break;
		}

		//extend the graph
		for (int j = 0; j < i.second.size(); j++) {
			auto exd_graph_ids = this->database->ReadVisualIDs(i.second[j]);
			int total_num_feats_j = exd_graph_ids.ids.rows();

			helper::ExtractTopFeatures(&(this->database->ReadKeypoints(i.second[j])), &exd_graph_ids, fileManager::parameters::maxNumFeatures);
			auto exdmatches = idsToMatches(exd_graph_ids, i.second[j]);

			igraph_t exd_graph;
			std::vector<cv::KeyPoint> pts;
			graph::buildFull(exdmatches, pts, exd_graph);

			if (GAN(&exd_graph, "n_vertices") == 0) {
				std::cerr << "nbhdGraph.Next: error: extend graph is empty!\n";
				break;
			}

			std::vector<cv::DMatch> twoViewMatches;

			//two view geometry includes all possible matches that some pairs id may exceed the maxNumFeat.
			auto matchFeatId = database->ReadTwoViewGeometry(i.first, i.second[j]).inlier_matches;

			//filter two view matches
			std::vector < cv::DMatch> newMatches;
			newMatches.reserve(matchFeatId.size());
			if (i.first < i.second[j]) {
				int idx1_limit = total_num_feats_i - fileManager::parameters::maxNumFeatures;
				int idx2_limit = total_num_feats_j - fileManager::parameters::maxNumFeatures;

				idx1_limit = idx1_limit > 0 ? idx1_limit : 0;
				idx2_limit = idx2_limit > 0 ? idx2_limit : 0;
				
				for (auto m : matchFeatId) {
					if (int(m.point2D_idx1) > idx1_limit && int(m.point2D_idx2) > idx2_limit) {
						newMatches.push_back(cv::DMatch(int(m.point2D_idx2) - idx2_limit, int(m.point2D_idx1)
							- idx1_limit, 0));
					}
				}
			}
			else
			{
				int idx1_limit = total_num_feats_j - fileManager::parameters::maxNumFeatures;
				int idx2_limit = total_num_feats_i - fileManager::parameters::maxNumFeatures;
				idx1_limit = idx1_limit > 0 ? idx1_limit : 0;
				idx2_limit = idx2_limit > 0 ? idx2_limit : 0;
				
				for (auto m : matchFeatId) {
					if (int(m.point2D_idx1) > idx1_limit && int(m.point2D_idx2) > idx2_limit) {
						newMatches.push_back(cv::DMatch(int(m.point2D_idx1) - idx1_limit, int(m.point2D_idx2)
							- idx2_limit, 0));
					}
				}
			}
			
			if (!graph::extend1to2(base_graph, exd_graph, newMatches)) {
				std::cerr << "nbhdGraph: error: graph extention failed! with img id "<<i.first<<"\n";
				break;
			}
			igraph_destroy(&exd_graph);
		}
		//do comparison with the base graph and stores the score
		std::vector<igraph_t> database_graph;
		database_graph.push_back(base_graph);
		auto candidate_scores = kernelobj.robustKernelCompWithQueryArray(database_graph);
		scores[next_index_].insert({i.first,candidate_scores[0][0]});
		igraph_destroy(&base_graph);
		std::cout << "\nNumber " << gCount << " th image extention tested";
		gCount++;
	}

	//write the result
	for (auto i : scores[next_index_]) {
		this->write_stream_ << this->query_image_names[next_index_] << "," << i.first << "," << i.second << "\n";
	}
	if (next_index_ % 10 == 0) {
		this->write_stream_.flush();
	}
	next_index_++;
	return -1;
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