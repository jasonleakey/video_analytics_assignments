/*
 * CaptureVideo.cpp
 *
 *  Created on: Jan 22, 2014
 *      Author: jasonleakey
 */

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <stdio.h>
#include <cv.h>
#include <math.h>

using namespace std;
using namespace cv;

// best color filter for my roommate's green apple. :)
static int lowerH = 0;
static int lowerS = 136;
static int lowerV = 60;

static int upperH = 120;
static int upperS = 256;
static int upperV = 256;
static Mat frame;
Mat markedImg;
static int fileNameCount = 0;

static void onClick(int event, int x, int y, int flags, void* userdata) {
	if (event == EVENT_LBUTTONDOWN) {
		// save one original video image and one object-marked image.
		stringstream ss;
		ss << "origin_";
		ss << fileNameCount;
		ss << ".jpg";
		imwrite(ss.str(), frame);
		ss.str("");
		ss << "marked_";
		ss << fileNameCount;
		ss << ".jpg";
		imwrite(ss.str(), markedImg);
		fileNameCount++;
	}
}

int main(int argc, char *argv[]) {
	VideoCapture capture("2014-05-07-103938.webm"); // capture from video device #0
//	VideoCapture capture("2014-05-04-115055.webm"); // capture from video device #0
	// the window name of original camera video.
	const string camWinName = "Camera Video";
	// the window name of marked apple video.
	const string markWinName = "Masked Video";
	// the file name of original camera video.
	const string origFilename = "apple_original.avi";
	// the file name of marked apple video
	const string markFilename = "apple_marked.avi";
	// check if the camera opens correctly.
	if (!capture.isOpened()) {
		cout << "Can not open the camera." << endl;
		return -1;
	}

	// read a frame to retrieve information.
	capture >> frame;
	// save as the "avi" codec, 18fps?
	VideoWriter video(origFilename, CV_FOURCC('X', 'V', 'I', 'D'), 18,
			frame.size());
	VideoWriter video2(markFilename, CV_FOURCC('X', 'V', 'I', 'D'), 18,
			frame.size());
	// check if the file opens correctly.
	if (!video.isOpened() || !video2.isOpened()) {
		cout << "VideoWriter has created." << endl;
	}

	// create two windows for showing real-time videos.
	namedWindow(camWinName, 1);
	namedWindow(markWinName, 1);
	// create trackbars to adjust color filter so we can detect the object we want.
	createTrackbar("Lower Hue", camWinName, &lowerH, 180, NULL);
	createTrackbar("Upper Hue", camWinName, &upperH, 180, NULL);

	createTrackbar("Lower Sat", camWinName, &lowerS, 256, NULL);
	createTrackbar("Upper Sat", camWinName, &upperS, 256, NULL);

	createTrackbar("Lower Val", camWinName, &lowerV, 256, NULL);
	createTrackbar("Upper Val", camWinName, &upperV, 256, NULL);
	// two windows links to the same callback, wherein left clicking saves the images.
	setMouseCallback(camWinName, onClick, NULL);
	setMouseCallback(markWinName, onClick, NULL);
	// recording until key 'q' or 'Esc' is pressed.
	while (true) {
		capture >> frame;
		if (frame.empty()) {
			cout << "Cannot get frame from the capture." << endl;
			break;
		}

		Mat framecopy;
		// convert the color space from RGB to HSV
		cvtColor(frame, framecopy, CV_BGR2HSV); //Change the color format from BGR to HSV
		// color detection of the object.
		inRange(framecopy, Scalar(lowerH, lowerS, lowerV),
				Scalar(upperH, upperS, upperV), markedImg);

		// remove noise by median filtering, 5x5 window.
		medianBlur(markedImg, markedImg, 5);
		// remove noises further by eroding&dilating.
		// use a 3x3 rectangle structuring element.
		// iterate twice for better result.
		erode(markedImg, markedImg, Mat(), Point(-1, -1), 2);

		// fill holes inside the apple
		// iterates 6 dilations and 2 erosions
		dilate(markedImg, markedImg, Mat(), Point(-1, -1), 10);
		erode(markedImg, markedImg, Mat(), Point(-1, -1), 2);

//		dilate(markedImg, markedImg, Mat(), Point(-1, -1), 10);
//        Mat markedImgCopy(markedImg);
//        cvtColor(markedImg, markedImgCopy, CV_GRAY2BGR);
//        cvtColor(markedImgCopy, markedImgCopy, CV_BGR2HSV);
//        bitwise_and(framecopy, markedImgCopy, framecopy);
		// save this frame to file.
		video << frame;
		cvtColor(markedImg, markedImg, CV_GRAY2BGR);
		video2 << markedImg;
		// update the windows
		imshow(camWinName, frame);
		imshow(markWinName, markedImg);
		// wait for keypress, 5ms delay.
		int key = waitKey(200);
		// Exit if Keypress 'q' or 'Esc'
		if (key == 'q' || key == 27) {
			break;
		}
	}
}
