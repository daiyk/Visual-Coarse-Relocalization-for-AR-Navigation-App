#include "nbhdGraph.h"
#include <StaticVRImg/extractor.h>
#include <StaticVRImg/graph.h>
#include <colmap/util/endian.h>
#include <fstream>

cv::Mat bitmapToMat(colmap::Bitmap &bitmap) {
	cv::Mat imgBitmap(bitmap.Height(), bitmap.Width(), CV_8U);
	auto dataFloat = bitmap.ConvertToColMajorArray();

	//transform bitmap to opencv image
	for (int i = 0; i < bitmap.Height(); i++) {
		for (int j = 0; j < bitmap.Width(); j++) {
			imgBitmap.at<uchar>(j, i) = dataFloat[j + i * bitmap.Width()];
		}
	}

	return imgBitmap;
}

nbhd::nbhdGraph::nbhdGraph():database(nullptr) {

}
nbhd::nbhdGraph::nbhdGraph(const fileManager::covisOptions& options) {
	//open the database
	database->Open(options.database_path);

	//scan the image list as query image
	this->query_image_names = options.image_list;

	//we use each image as query unit
	this->query_images = this->Read(options.image_list);

	//read dictionary and build kdtree
	this->vocab = std::make_unique<matcher::kdTree>(readVocab(options.vocab_path));
	this->vocab_path_ = options.vocab_path;

	//init covismap
	this->map = std::make_unique<kernel::covisMap>(vocab->numWords(),database->NumImages());

	//construct the query graphs and store it
	this->preprocess();
		
}
nbhd::nbhdGraph::~nbhdGraph() {
	 database->Close();
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
		query_images.push_back(bitmapToMat(bitimg).clone());
	}
	return query_images;
}

//vocabtree matching from query image features and build graphs on top of it 
void nbhd::nbhdGraph::preprocess() {
	//extract features from the query and build graphs
	this->query_graphs.clear();
	this->query_graphs.resize(NumQuery());
	for (int i = 0; i < NumQuery(); i++) {
		auto qry_img =this->query_images[i];
		std::vector<cv::KeyPoint> points;
		cv::Mat descripts;
		extractor::vlimg_descips_compute_simple(qry_img, descripts, points);

		//matching with dictionary
		auto matches = this->vocab->colmapSearch(descripts);

		//build graph on top of the matches
		igraph_t graph;
		graph::buildFull(matches, points, graph);

		//stores the query graph for future comparison
		igraph_copy(&query_graphs[i], &graph);
	}
	//build invert_index_
	for (auto i : database->ReadAllImages()) {
		auto ids = database->ReadVisualIDs(i.ImageId());
		map->addEntry(i.ImageId(),ids);
	}
	//if preprocess success, set the next_index_
	this->next_index_ = 0;
}

cv::Mat nbhd::nbhdGraph::getVocab() {
	return this->vocab->getVocab();
}

void nbhd::nbhdGraph::setVocab(std::string vocabPath) {
	this->vocab.reset();
	this->vocab = std::make_unique<matcher::kdTree>(readVocab(vocabPath));
}

void nbhd::nbhdGraph::setVocab(cv::Mat vocab) {
	this->vocab.reset();
	this->vocab = std::make_unique<matcher::kdTree>(vocab);
}
cv::Mat nbhd::nbhdGraph::readVocab(std::string vocabPath) {
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
	colmap::FeatureVisualIDs qry_id(visual_id, this->vocab->numWords(), this->vocab_path_);

	//query the database and get candidate matching images
	std::vector<int> candidates;
	this->map->Query(qry_id, candidates);

	//query database and do graph extention based on covisibility
	database->Read

}