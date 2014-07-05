#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <cstring>
#include <cctype>
#include <stdio.h>
#include <cv.h>
#include <math.h>

using namespace cv;
using namespace std;

static const bool DEBUG = true;

static const bool SHOW_MATCHED_IMAGE = true;
static Mat frameTemplate;
static Mat frame;
// static void help()
// {
//     printf(
//             "\nDemonstrate the use of the HoG descriptor using\n"
//             "  HOGDescriptor::hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());\n"
//             "Usage:\n"
//             "./peopledetect (<image_filename> | <image_list>.txt)\n\n");
// }

static void temocTracking(Mat frameScene) {
	if (frameTemplate.empty() || frameScene.empty()) {
		return;
	}

	Mat frame_object;
	Mat frame_scene;

	if (DEBUG) {
		cout << "aaaaa" << endl;
	}
	// the algorithm works on GREYSCALE.
	cvtColor(frameTemplate, frame_object, CV_RGB2GRAY);
	cvtColor(frameScene, frame_scene, CV_RGB2GRAY);

	// Use SURF alorithm.
	int minHessian = 400;
	//	SiftFeatureDetector detector;
	SurfFeatureDetector detector(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2;

	// detect keypoints
	detector.detect(frame_object, keypoints_1);
	detector.detect(frame_scene, keypoints_2);

	Mat img_keypoints_1, img_keypoints_2;

	if (SHOW_MATCHED_IMAGE) {
		// draw keypoints in memory.
		drawKeypoints(frame_object, keypoints_1, img_keypoints_1,
				Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		drawKeypoints(frame_scene, keypoints_2, img_keypoints_2,
				Scalar::all(-1), DrawMatchesFlags::DEFAULT);

		imshow("sift_keypoints_1", img_keypoints_1);
		imshow("sift_keypoints_2", img_keypoints_2);
	}

	if (keypoints_1.empty() || keypoints_2.empty()) {
		if (DEBUG) cerr << "NO KEYPOINTS!" << endl;
		return;
	}

	// use surf?
	//SiftDescriptorExtractor extractor;
	SurfDescriptorExtractor extractor;

	Mat descriptors_1, descriptors_2;

	// compute the descriptors from keypoints
	extractor.compute(frame_object, keypoints_1, descriptors_1);
	extractor.compute(frame_scene, keypoints_2, descriptors_2);

	if (descriptors_1.empty() || descriptors_2.empty()) {
		if (DEBUG) cerr << "NO DESCRIPTORS!" << endl;
		return;
	}

	double max_dist = 0;
	double min_dist = 100;

	FlannBasedMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	// Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_1.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist)
			min_dist = dist;
		if (dist > max_dist)
			max_dist = dist;
	}

	// Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	vector<DMatch> good_matches;
	for (int i = 0; i < descriptors_1.rows; i++) {
		if (matches[i].distance < 3 * min_dist) {
			good_matches.push_back(matches[i]);
		}
	}

	Mat img_matches;
	if (SHOW_MATCHED_IMAGE) {
		drawMatches(frame_object, keypoints_1, frame_scene, keypoints_2,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	}

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++) {
		// Get the keypoints from the good matches
		obj.push_back(keypoints_1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_2[good_matches[i].trainIdx].pt);
	}

	// too few matches to get the homography. ignore.
	if (good_matches.size() < 4) {
		return;
	}

	// find the transform between matched keypoints.
	Mat H = findHomography(obj, scene, CV_RANSAC);

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0);
	obj_corners[1] = cvPoint(frame_object.cols, 0);
	obj_corners[2] = cvPoint(frame_object.cols, frame_object.rows);
	obj_corners[3] = cvPoint(0, frame_object.rows);
	std::vector<Point2f> scene_corners(4);

	// Performs the perspective matrix transformation of vectors.
	perspectiveTransform(obj_corners, scene_corners, H);

	if (SHOW_MATCHED_IMAGE) {
		//	Draw lines between the corners (the mapped object in the scene - image_2 )
		line(img_matches, scene_corners[0] + Point2f(frame_object.cols, 0),
				scene_corners[1] + Point2f(frame_object.cols, 0),
				Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[1] + Point2f(frame_object.cols, 0),
				scene_corners[2] + Point2f(frame_object.cols, 0),
				Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[2] + Point2f(frame_object.cols, 0),
				scene_corners[3] + Point2f(frame_object.cols, 0),
				Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[3] + Point2f(frame_object.cols, 0),
				scene_corners[0] + Point2f(frame_object.cols, 0),
				Scalar(0, 255, 0), 4);
	}

	//-- Draw lines between the corners in output iamge
	line(frame, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
	line(frame, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
	line(frame, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
	line(frame, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);

	if (SHOW_MATCHED_IMAGE) {
		// show the line segments
		namedWindow("Surf Matches", 0);
		imshow("Surf Matches", img_matches);
	}
}

int main(int argc, char** argv) {
	Mat img;
	FILE* f = 0;
	char _filename[1024];
//	VideoCapture capture("2014-04-12 21.22.05.mp4");
//	VideoCapture capture("2014-04-12 21.23.42.mp4");
	VideoCapture capture("2014-05-04-115055.webm");
//	VideoCapture capture("2014-04-12 21.20.28.mp4");
//	frameTemplate = imread("2014-04-12 21.24.02_300x128.jpg");


	if (argc > 1 && false) {
		img = imread(argv[1]);

		if (img.data) {
			strcpy(_filename, argv[1]);
		} else {
			f = fopen(argv[1], "rt");
			if (!f) {
				fprintf(stderr,
						"ERROR: the specified file could not be loaded\n");
				return -1;
			}
		}
	} else {
		if (!capture.isOpened()) {
			cout << "Webcam cannot be opened!" << endl;
			return 0;
		}
	}

	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	namedWindow("people detector", WINDOW_NORMAL);

	for (;;) {
		if (argc > 1 && false) {
			char* filename = _filename;
			if (f) {
				if (!fgets(filename, (int) sizeof(_filename) - 2, f))
					break;
				//while(*filename && isspace(*filename))
				//  ++filename;
				if (filename[0] == '#')
					continue;
				int l = (int) strlen(filename);
				while (l > 0 && isspace(filename[l - 1]))
					--l;
				filename[l] = '\0';
				img = imread(filename);
			}
			printf("%s:\n", filename);
			if (!img.data)
				continue;
		} else {
			capture >> img;
			if (img.empty()) {
				cerr << "Empty frame. " << endl;
				return 0;
			}
		}

		cout << img.size() << endl;
		float scale = 1;
		int w = img.size().width * scale;
		int h = img.size().height * scale;
		resize(img, img, Size(w, h), 0, 0, CV_INTER_AREA);
//		transpose(img, img);
//		flip(img, img, 1);
		cout << img.size() << endl;

		fflush(stdout);
		vector<Rect> found, found_filtered;
		double t = (double) getTickCount();
		// run the detector with default parameters. to get a higher hit-rate
		// (and more false alarms, respectively), decrease the hitThreshold and
		// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
		hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
		t = (double) getTickCount() - t;
		printf("\tdetection time = %gms\n", t * 1000. / cv::getTickFrequency());
		cout << "aaa" << endl;
		size_t i, j;
		for (i = 0; i < found.size(); i++) {
			Rect r = found[i];
			for (j = 0; j < found.size(); j++)
				if (j != i && (r & found[j]) == r)
					break;
			if (j == found.size())
			found_filtered.push_back(r);
		}
		cout << "bbb" << endl;

		Mat imgCopy = img.clone();
		Rect max;
		int idx = -1;
		for (i = 0; i < found_filtered.size(); i++) {
			Rect r = found_filtered[i];
			// the HOG detector returns slightly larger rectangles than the real objects.
			// so we slightly shrink the rectangles to get a nicer output.
			r.x += cvRound(r.width * 0.1);
			r.width = cvRound(r.width * 0.8);
			r.y += cvRound(r.height * 0.07);
			r.height = cvRound(r.height * 0.8);

			rectangle(img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
			if (r.area() > max.area()) {
				max = r;
				idx = i;
			}
		}
		cout << "ccc" << endl;
		if (idx >= 0) {
			frame = imgCopy(max & Rect(0, 0, img.cols, img.rows));
			cout << max.x << "," << max.y << ";" << max.width << "," << max.height
					<< endl;
			temocTracking(imgCopy);
		}
		cout << "ddd" << endl;
		imshow("people detector", img);
		int c = waitKey(20) & 255;
		if (c == 'q' || c == 'Q')
			break;
	}
	if (f)
		fclose(f);
	return 0;
}
