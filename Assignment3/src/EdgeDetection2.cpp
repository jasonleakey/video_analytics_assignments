/*
 * CaptureVideo.cpp
 *
 *  Created on: Jan 22, 2014
 *      Author: jasonleakey
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

// best color filter for my roommate's green apple. :)
static int lowerH = 24;
static int lowerS = 100;
static int lowerV = 40;

static int upperH = 80;
static int upperS = 256;
static int upperV = 256;

static Mat frame;
static Mat framecopy;
static Mat markedImg;
static int fileNameCount = 0;
static int thresh = 9;
static int max_thresh = 255;
static int dp = 1;
static int max_dp = 2;
static int minDist = 2000;
static int max_minDist = 2000;
static int param1 = 20;
static int max_param1 = 2000;
static int param2 = 20;
static int max_param2 = 2000;
static int minRadius = 60;
static int max_minRadius = 2000;
static int maxRadius = 70;
static int max_maxRadius = 2000;
static RNG rng(12345);
int ratio = 3;
int kernel_size = 3;

/// Function header
void thresh_callback(int, void*);
void CannyThreshold(int, void*);

int main(int argc, char *argv[]) {
	// the window name of original apple video.
	const string origWinName = "Original Video";
	// the file name of original camera video.
	const string origFilename = "apple_original.avi";
	VideoCapture capture(origFilename);
	// the file name of marked apple video
	const string trackedFilename = "apple_marked.avi";

	if (!capture.isOpened()) {
		cout << "VideoCapture opening failed." << endl;
		return -1;
	}
	cout << "FPS:" << capture.get(CV_CAP_PROP_FPS) << endl;

	// save as the "avi" codec, 18fps?
	VideoWriter video(trackedFilename, CV_FOURCC('X', 'V', 'I', 'D'), 18,
			frame.size());
	// check if the file opens correctly.
	if (!video.isOpened()) {
		cout << "VideoWriter cannot be created." << endl;
	}

	namedWindow(origWinName, CV_WINDOW_AUTOSIZE);
	/// Create a Trackbar for user to enter threshold
	createTrackbar("Min Threshold:", origWinName, &thresh, max_thresh,
			CannyThreshold);
	createTrackbar("dp:", origWinName, &dp, max_dp, NULL);
	createTrackbar("minDist:", origWinName, &minDist, max_minDist, NULL);
	createTrackbar("param1:", origWinName, &param1, max_param1, NULL);
	createTrackbar("param2:", origWinName, &param2, max_param2, NULL);
	createTrackbar("minRadius:", origWinName, &minRadius, max_minRadius, NULL);
	createTrackbar("maxRadius:", origWinName, &maxRadius, max_minRadius, NULL);
	namedWindow("Edge Map", CV_WINDOW_AUTOSIZE);
//	namedWindow("Binary Image", CV_WINDOW_AUTOSIZE);
//	createTrackbar(" Threshold:", origWinName, &thresh, max_thresh,
//			thresh_callback);

	while (true) {
		// play the original apple video.
		capture >> frame;
		if (frame.empty()) {
			cout << "Cannot get frame from the capture." << endl;
			break;
		}

		/// Convert image to gray and blur it
		cvtColor(frame, framecopy, CV_BGR2GRAY);
//		blur(framecopy, framecopy, Size(3, 3));
		// smooth it, otherwise a lot of false circles may be detected
		GaussianBlur(framecopy, framecopy, Size(9, 9), 2, 2);

//		imshow("Binary Image", framecopy);
		  /// Show the image
		  CannyThreshold(0, 0);
		vector<Vec3f> circles;
		HoughCircles(framecopy, circles, CV_HOUGH_GRADIENT, dp,
				minDist, param1, param2, minRadius, maxRadius);
		for (size_t i = 0; i < circles.size(); i++) {
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
//			 draw the circle center
			circle(frame, center, 3, Scalar(0, 255, 0), -1, 8, 0);
//			 draw the circle outline
			circle(frame, center, radius, Scalar(0, 0, 255), 3, 8, 0);
		}
//		thresh_callback(0, 0);

		imshow(origWinName, frame);
		// wait for keypress, 5ms delay.
		int key = waitKey(80);
		// Exit if Keypress 'q' or 'Esc'
		if (key == 'q' || key == 27) {
			break;
		}
	}
	// BGR ==> HSV
	// color filtering
	// dilate & erode to remove holes.
	// bilateral filtering to remove noise.
	// edge detection (circle?)
	// draw a box around the apple from the origin window.

	// save this frame to file.
}

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void CannyThreshold(int, void*) {
	/// Reduce noise with a kernel 3x3
	blur(framecopy, framecopy, Size(3, 3));

	/// Canny detector
	Canny(framecopy, framecopy, thresh, thresh * ratio,
			kernel_size);

	/// Using Canny's output as a mask, we display our result
	Mat dst = (Mat) Scalar::all(0);

	frame.copyTo(dst, framecopy);
	imshow("Edge Map", dst);
}

/** @function thresh_callback */
void thresh_callback(int, void*) {
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using Threshold
	threshold(framecopy, threshold_output, thresh, 255, THRESH_BINARY);
	/// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE,
			CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f> center(contours.size());
	vector<float> radius(contours.size());

	for (int i = 0; i < contours.size(); i++) {
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		minEnclosingCircle((Mat) contours_poly[i], center[i], radius[i]);
	}

	/// Draw polygonal contour + bonding rects + circles
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
				rng.uniform(0, 255));
		drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0,
				Point());
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8,
				0);
		rectangle(frame, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
		circle(drawing, center[i], (int) radius[i], color, 2, 8, 0);
	}

	/// Show in a window
	imshow("Contours", drawing);
}
