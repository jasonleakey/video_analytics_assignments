/*
 * EdgeDetection.cpp
 *
 *  Created on: Jan 22, 2014
 *      Author: Yetian Huang
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
static int fileNameCount = 0;
static int thresh = 14;
int ratio = 3;
int kernel_size = 3;

void edge_detect_callback(int, void*);
static void onClick(int event, int x, int y, int flags, void* userdata);

int main(int argc, char *argv[]) {
	// the window name of original apple video.
	const string origWinName = "Original Video";
	const string edgeWinName = "Edge and Box";
	const string binImgWinName = "Binary Image";
	// the file name of original camera video.
	const string origFilename = "apple_original.avi";
	VideoCapture capture(origFilename);
	// the file name of marked apple video
	const string trackedFilename = "apple_tracked.avi";

	if (!capture.isOpened()) {
		cout << "VideoCapture opening failed." << endl;
		return -1;
	}
	// read a frame to retrieve information.
	capture >> frame;
	// save as the "avi" codec, 18fps?
	VideoWriter video(trackedFilename, CV_FOURCC('X', 'V', 'I', 'D'), 18,
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
	setMouseCallback(binImgWinName, onClick, NULL);

	while (true) {
		// play the original apple video.
		capture >> frame;
		if (frame.empty()) {
			cout << "No more frames from the capture." << endl;
			break;
		}

		// convert the color space from RGB to HSV
		cvtColor(frame, framecopy, CV_BGR2HSV); //Change the color format from BGR to HSV
		// color detection of the object.
		inRange(framecopy, Scalar(lowerH, lowerS, lowerV),
				Scalar(upperH, upperS, upperV), framecopy);

		// remove noise by median filtering, 5x5 window.
		medianBlur(framecopy, framecopy, 5);
		// remove small noises outside the apple by eroding->dilating.
		// use a 3x3 rectangle structuring element.
		// iterate twice for better result.
		erode(framecopy, framecopy, Mat(), Point(-1, -1), 2);
		dilate(framecopy, framecopy, Mat(), Point(-1, -1), 2);

		// fill holes inside the apple
		// iterates 6 dilations and 2 erosions
		dilate(framecopy, framecopy, Mat(), Point(-1, -1), 6);
		erode(framecopy, framecopy, Mat(), Point(-1, -1), 2);

		imshow(binImgWinName, framecopy);

		edge_detect_callback(0, 0);

		imshow(origWinName, frame);
		video << frame;
		// wait for keypress, 80ms delay.
		// make the video play slower.
		int key = waitKey(80);
		// Exit if Keypress 'q' or 'Esc'
		if (key == 'q' || key == 27) {
			break;
		}
	}
}

static void onClick(int event, int x, int y, int flags, void* userdata) {
	if (event == EVENT_LBUTTONDOWN) {
		// save one original video image and one object-marked image.
		stringstream ss;
		ss << "tracked_frame_";
		ss << fileNameCount;
		ss << ".jpg";
		imwrite(ss.str(), frame);
		fileNameCount++;
	}
}

void edge_detect_callback(int, void*) {
	Mat detected_edge;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	// Detect edges using Canny
	Canny(framecopy, detected_edge, thresh, thresh * ratio,
			kernel_size);
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

	// Draw polygonal contour + bounding rects
	for (int i = 0; i < contours.size(); i++) {
		// the yellow contour
		drawContours(frame, contours_poly, i, Scalar(0, 255, 255), 1, 8, vector<Vec4i>(), 0,
				Point());
		// draw bounding rect
		rectangle(frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 2, 8,
				0);
		// draw bounding rect
		rectangle(frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 2, 8,
				0);
	}
}
