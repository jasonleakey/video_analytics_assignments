/*
 * ObjectTracking.cpp
 *
 *  Created on: Jan 22, 2014
 *      Author: Yetian Huang
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include "opencv2/core/core.hpp"//因为在属性中已经配置了opencv等目录，所以把其当成了本地目录一样
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

// best color filter for my roommate's green apple. :)
static int lowerH = 24;
static int lowerS = 90;
static int lowerV = 40;

static int upperH = 80;
static int upperS = 256;
static int upperV = 256;

static const bool ENABLE_VELOCITY = false;
static const bool SHOW_MATCHED_IMAGE = false;

static VideoCapture capture;
static Mat origFrame;
static Mat frame;
static Mat framecopy;
static Mat frameTemplate;
static int fcount = 0;
// the FPS for my camera.
static const int DEFAULT_FPS = 18;
static int cam_fps = DEFAULT_FPS;
static long double vel = 0.0;
static int thresh = 14;
int ratio = 3;
int kernel_size = 3;

void edge_detect_callback(int, void*);
static void onClick(int event, int x, int y, int flags, void* userdata);
static void temocTracking(Mat frameScene);

int main(int argc, char *argv[]) {
	// the file name of original camera video.
	string origFilename;
	frameTemplate = imread("./data/train/188.temoc/01.jpg");
	if (argc == 2) {
		origFilename = argv[1];
		capture.open(origFilename);
	} else if (argc == 1) {
		capture.open(0);
		// does not work well on my Linux system
//		cam_fps = (int) capture.get(CV_CAP_PROP_FPS);
	} else {
		return -1;
	}

	// the window name of original apple video.
	const string origWinName = "Original Video";
	const string edgeWinName = "Edge and Box";
	const string binImgWinName = "Binary Image";
	// the file name of marked apple video
	const string trackedFilename = "object_tracking.avi";

	if (!capture.isOpened()) {
		cout << "VideoCapture opening failed." << endl;
		return -1;
	}
	// read a frame to retrieve information.
	capture >> frame;
	// save as the "avi" codec, 18fps?
	VideoWriter video(trackedFilename, CV_FOURCC('X', 'V', 'I', 'D'), cam_fps,
			frame.size());
	// check if the file opens correctly.
	if (!video.isOpened()) {
		cout << "VideoWriter cannot be created." << endl;
		return -1;
	}

	namedWindow(origWinName, CV_WINDOW_AUTOSIZE);
	namedWindow(binImgWinName, CV_WINDOW_AUTOSIZE);
	moveWindow(binImgWinName, 700, 0);
	setMouseCallback(origWinName, onClick, NULL);
//	setMouseCallback(binImgWinName, onClick, NULL);
//	createButton(nameBut1, onButtonClick);

	while (true) {
		// play the original apple video.
		capture >> origFrame;
		if (origFrame.empty()) {
			cout << "No more frames from the capture." << endl;
			break;
		}
		frame = origFrame.clone();

		// convert the color space from RGB to HSV
		cvtColor(origFrame, framecopy, CV_BGR2HSV); //Change the color format from BGR to HSV
		// color detection of the object.
		inRange(framecopy, Scalar(lowerH, lowerS, lowerV),
				Scalar(upperH, upperS, upperV), framecopy);

		// remove noise by median filtering, 5x5 window.
		medianBlur(framecopy, framecopy, 5);
		// remove small noises outside the apple by eroding->dilating.
		// use a 3x3 rectangle structuring element.
		// iterate twice for better result.
		erode(framecopy, framecopy, Mat(), Point(-1, -1), 12);

		// fill holes inside the apple
		// iterates 6 dilations and 2 erosions
		dilate(framecopy, framecopy, Mat(), Point(-1, -1), 10);
		erode(framecopy, framecopy, Mat(), Point(-1, -1), 2);

		imshow(binImgWinName, framecopy);

		temocTracking(origFrame);

		edge_detect_callback(0, 0);

		imshow(origWinName, frame);
		video << frame;
		// wait for keypress, 80ms delay.
		// make the video play slower.
		int key = waitKey(40);
		// Exit if Keypress 'q' or 'Esc'
		if (key == 'q' || key == 27) {
			break;
		}
	}
}

static void temocTracking(Mat frameScene) {
	if (frameTemplate.empty() || frameScene.empty()) {
		return;
	}

	Mat frame_object;
	Mat frame_scene;
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

	// use surf?
	//SiftDescriptorExtractor extractor;
	SurfDescriptorExtractor extractor;

	Mat descriptors_1, descriptors_2;

	// compute the descriptors from keypoints
	extractor.compute(frame_object, keypoints_1, descriptors_1);
	extractor.compute(frame_scene, keypoints_2, descriptors_2);

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
		imshow("Surf Matches", img_matches);
	}
}

//Function to find the velocity of the object, not used for this assignment 4
static double updateVelocity(int x2, int y2) {
	//Increment the frame
	fcount++;
	//Variables
	static double dis = 0;
	double dis1 = 0;
	long double x, y;
	// Holding the last and current positions
	static int x1 = 0;
	static int y1 = 0;
	// valid points
	if (x1 > 1 && x2 > 1 && y1 > 1 && y2 >> 1) {
		x = (x2 - x1);
		x = x * x;
		y = (y2 - y1);
		y = y * y;
		dis1 = sqrt(x + y);
	}
	// accumulate the total distance.
	dis = dis + dis1;
	if (fcount == 15) {
		// velocity = distance / (past time);
		vel = dis / (1.0 / cam_fps * 15);
		//Cleanup
		fcount = 0;
		dis = 0;
	}

	x1 = x2;
	y1 = y2;

	return vel;
}

// When the window is clicked. we record a frame as the object/template.
static void onClick(int event, int x, int y, int flags, void* userdata) {
//	if (event == EVENT_LBUTTONDOWN) {
//		// save one original video image and one object-marked image.
//		stringstream ss;
//		ss << "tracked_frame_";
//		ss << fileNameCount;
//		ss << ".jpg";
//		imwrite(ss.str(), frame);
//		fileNameCount++;
//	}
	if (event == EVENT_LBUTTONDOWN) {
		string templateWinName = "Template";
//		imshow(templateWinName, origFrame);
		frameTemplate = origFrame.clone();
	}
}

// detect the ball edge.
void edge_detect_callback(int, void*) {
	Mat detected_edge;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	// Detect edges using Canny
	Canny(framecopy, detected_edge, thresh, thresh * ratio, kernel_size);
	// Find contours
	// ideally, we only get one contour, i.e. the apple.
	findContours(detected_edge, contours, hierarchy, CV_RETR_TREE,
			CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f> center(contours.size());
	vector<float> radius(contours.size());

	// get bounding rectangles around the polygons.
	for (int i = 0; i < contours.size(); i++) {
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}

	// Finds the contour with the largest area
	int area = 0;
	int idx = -1;
	for (int i = 0; i < contours.size(); i++) {
		int s = boundRect[i].width * boundRect[i].height;
		if (s > area) {
			idx = i;
			area = s;
		}
	}

	if (idx >= 0) {
		// we ONLY draw the largest polygonal contour + bounding rect
		drawContours(frame, contours_poly, idx, Scalar(0, 255, 255), 1, 8,
				vector<Vec4i>(), 0, Point());
		// draw bounding rect
		rectangle(frame, boundRect[idx].tl(), boundRect[idx].br(),
				Scalar(0, 0, 255), 2, 8, 0);
		// draw bounding rect
		rectangle(frame, boundRect[idx].tl(), boundRect[idx].br(),
				Scalar(0, 0, 255), 2, 8, 0);

		if (ENABLE_VELOCITY) {
			// calculate velocity by tracing the movement of center of bounding rect.
			updateVelocity(boundRect[idx].x + boundRect[idx].width / 2,
					boundRect[idx].y + boundRect[idx].height / 2);

			// add velocity to frame.
			stringstream ss;
			ss << vel;
			putText(frame, ss.str() + " px/s", Point(30, 30),
					FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(0, 0, 255), 1,
					CV_AA);
		}
	}
}
