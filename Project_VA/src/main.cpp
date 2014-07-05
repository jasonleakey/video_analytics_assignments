#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
using namespace std;
using namespace cv;

//location of the training data
char *TRAINING_DATA_DIR = "train.txt";
//location of the evaluation data
char *EVAL_DATA_DIR = "test.txt";

//See article on BoW model for details
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");

//See article on BoW model for details
int dictionarySize = 1000;
TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;

//See article on BoW model for details
BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
//See article on BoW model for details
BOWImgDescriptorExtractor bowDE(extractor, matcher);

/**
 * \brief Recursively traverses a folder hierarchy. Extracts features from the training images and adds them to the bowTrainer.
 */
void extractTrainingVocabulary(char* indexFile) {
	FILE *fp = fopen(indexFile, "r");
	char filename[1024];
	int label;
	if (!fp) {
		cerr << "Error: Could not find training index file " << filename
				<< endl;
		return;
	}
	while (!feof(fp)) {
		fscanf(fp, "%d %s", &label, filename);

		cout << "Processing file " << filename << endl;
		Mat img = imread(filename);
		if (!img.empty()) {
			vector<KeyPoint> keypoints;
			detector->detect(img, keypoints);
			if (keypoints.empty()) {
				cerr << "Warning: Could not find key points in image: "
						<< filename << endl;
			} else {
				Mat features;
				extractor->compute(img, keypoints, features);
				bowTrainer.add(features);
			}
		} else {
			cerr << "Warning: Could not read image: " << filename << endl;
		}
	}
	fclose(fp);
}

/**
 * \brief Recursively traverses a folder hierarchy. Creates a BoW descriptor for each image encountered.
 */
void extractBOWDescriptor(char *indexFile, Mat& descriptors, Mat& labels) {
	FILE *fp = fopen(indexFile, "r");
	char filename[1024];
	float label;
	if (!fp) {
		cerr << "Error: Could not find training index file " << filename
				<< endl;
		return;
	}
	while (!feof(fp)) {
		fscanf(fp, "%f %s", &label, filename);
		cout << "Processing file " << filename << endl;
		Mat img = imread(filename);
		if (!img.empty()) {
			vector<KeyPoint> keypoints;
			detector->detect(img, keypoints);
			if (keypoints.empty()) {
				cerr << "Warning: Could not find key points in image: "
						<< filename << endl;
			} else {
				Mat bowDescriptor;
				bowDE.compute(img, keypoints, bowDescriptor);
				descriptors.push_back(bowDescriptor);
				labels.push_back(label);
			}
		} else {
			cerr << "Warning: Could not read image: " << filename << endl;
		}
	}
	fclose(fp);
}

int main(int argc, char ** argv) {
	cv::initModule_nonfree();
	cout << "Creating dictionary..." << endl;
	extractTrainingVocabulary(TRAINING_DATA_DIR);
	vector<Mat> descriptors = bowTrainer.getDescriptors();
	int count = 0;
	for (vector<Mat>::iterator iter = descriptors.begin();
			iter != descriptors.end(); iter++) {
		count += iter->rows;
	}
	cout << "Clustering " << count << " features" << endl;
	Mat dictionary = bowTrainer.cluster();
	bowDE.setVocabulary(dictionary);
	cout << "Processing training data..." << endl;
	Mat trainingData(0, dictionarySize, CV_32FC1);
	Mat labels(0, 1, CV_32FC1);
	extractBOWDescriptor(TRAINING_DATA_DIR, trainingData, labels);

	NormalBayesClassifier classifier;

	cout << "Training classifier..." << endl;

	classifier.train(trainingData, labels);

	cout << "Processing evaluation data..." << endl;
	Mat evalData(0, dictionarySize, CV_32FC1);
	Mat groundTruth(0, 1, CV_32FC1);
	extractBOWDescriptor(EVAL_DATA_DIR, evalData, groundTruth);

	cout << "Evaluating classifier..." << endl;
	Mat results;
	classifier.predict(evalData, &results);

	double errorRate = (double) countNonZero(groundTruth - results)
			/ evalData.rows;

	cout << "Error rate: " << errorRate << endl;
	getchar();
}
