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

using namespace cv;
using namespace std;

static const bool DEBUG = false;

// if two rectangles are closer than this threshold, we cluster them.
static const int CLOSE_RECT_THRESHOLD = 30;
static const bool DEBUG_SHOW_MATCHED_IMAGE = true;
static bool DEBUG_CLOSE_DISTANCE = false;
// the template image in SURF.
static Mat frameTemplate;
static Mat rawCopyForSelection;
static Mat rawCopyForDraw;
// the first point when mouse is pressed down.
static Point originSelPoint;
// the bounding rect of right person who wears UTDALLAS T-shirt.
static Rect keyPersonRect;
// the selected area/rectangle of template
static Rect selection;
// the ROI we selected by mouse is leveraged for SURF template
static Mat roi;
// the bounding polygon around UTDALLAS Logo.
static vector<Point2f> SURFMatchedPolygon(4);
// are we selecting area
static bool select_object = false;
// the frame processing is paused for selection.
static bool paused = false;
// should we start recognition yet?
static bool startRecognition = false;
// best color filter for the UTDALLAS Logo.
static int lowerH = 0;
static int lowerS = 136;
static int lowerV = 60;

static int upperH = 120;
static int upperS = 256;
static int upperV = 256;

static Mat filter_utdallas_color(Mat);

static void onClick(int event, int x, int y, int flags, void* param) {
	switch (event) {
    // middle button to pause/play
	case CV_EVENT_MBUTTONDOWN:
	{
		if (paused) {
            // change our template
            if (selection.width > 0 && selection.height > 0) {
            	frameTemplate = roi.clone();
                // after we copied ROI, clear selection.
                selection = Rect(0, 0, 0, 0);
            }
            startRecognition = true;
		}
        paused = !paused;
        break;
	}
    // left button pressed down
	case CV_EVENT_LBUTTONDOWN:
        originSelPoint = Point(x, y);
		selection = Rect(x, y, 0, 0);
		select_object = true;
		break;
    // left button up
	case CV_EVENT_LBUTTONUP:
        selection.width = abs(x - originSelPoint.x);
        selection.height = abs(y - originSelPoint.y);
        selection.x = x > originSelPoint.x ? originSelPoint.x : x;
        selection.y = y > originSelPoint.y ? originSelPoint.y : y;
		// turn off selection mode
		select_object = false;
		break;
	}
}

static void temocTracking(Mat frameScene) {
	if (frameTemplate.empty() || frameScene.empty()) {
		return;
	}

	Mat frame_object;
	Mat frame_scene;

    // only match in ROI.
	Mat mask = filter_utdallas_color(frameScene);
    Mat frameSceneMasked;
    frameScene.copyTo(frameSceneMasked, mask);

	if (DEBUG) {
//		cout << "aaaaa" << endl;
        imshow("masked", frameSceneMasked);
	}
	// the algorithm works on GREYSCALE.
	cvtColor(frameTemplate, frame_object, CV_RGB2GRAY);
	cvtColor(frameSceneMasked, frame_scene, CV_RGB2GRAY);

	// Use SURF alorithm.
	int minHessian = 400;
	//	SiftFeatureDetector detector;
	SurfFeatureDetector detector(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2;

	// detect keypoints
	detector.detect(frame_object, keypoints_1);
	detector.detect(frame_scene, keypoints_2);

	Mat img_keypoints_1, img_keypoints_2;

	if (DEBUG_SHOW_MATCHED_IMAGE) {
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
	if (DEBUG_SHOW_MATCHED_IMAGE) {
		drawMatches(frame_object, keypoints_1, frame_scene, keypoints_2,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	}

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (unsigned int i = 0; i < good_matches.size(); i++) {
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

	if (DEBUG_SHOW_MATCHED_IMAGE) {
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
    Point2f shift = Point2f(0, 0); // Deprecated: keyPersonRect.tl();
	line(rawCopyForDraw, scene_corners[0] + shift, scene_corners[1] + shift, Scalar(0, 255, 0), 4);
	line(rawCopyForDraw, scene_corners[1] + shift, scene_corners[2] + shift, Scalar(0, 255, 0), 4);
	line(rawCopyForDraw, scene_corners[2] + shift, scene_corners[3] + shift, Scalar(0, 255, 0), 4);
	line(rawCopyForDraw, scene_corners[3] + shift, scene_corners[0] + shift, Scalar(0, 255, 0), 4);

	SURFMatchedPolygon[0] = scene_corners[0] + shift;
	SURFMatchedPolygon[1] = scene_corners[1] + shift;
	SURFMatchedPolygon[2] = scene_corners[2] + shift;
	SURFMatchedPolygon[3] = scene_corners[3] + shift;

	if (DEBUG_SHOW_MATCHED_IMAGE) {
		// show the line segments
		namedWindow("Surf Matches", 0);
		imshow("Surf Matches", img_matches);
	}
}

// if two rectangles are close enough,
bool isCloseEnough(Rect a, Rect b) {
    Rect c = a & b;
    // a and b intersect. of course, they are close.
	if (c.area() != 0) {
        if (DEBUG_CLOSE_DISTANCE) {
        	cout << a << " and " << b << " are intersected." << endl;
        }
		return true;
	}

    int shortestDist = INT32_MAX;
    Point a1 = a.tl();
    Point a2 = Point(a.x + a.width, a.y);
    Point a3 = Point(a.x, a.y + a.height);
    Point a4 = a.br();
    Point b1 = b.tl();
    Point b2 = Point(b.x + b.width, b.y);
    Point b3 = Point(b.x, b.y + b.height);
    Point b4 = b.br();
	// case 1:
	if (b1.x > a4.x && b1.y > a4.y) {
		shortestDist = sqrt(pow(b1.x - a4.x, 2) + pow(b1.y - a4.y, 2));
        if (DEBUG_CLOSE_DISTANCE) {
        	cout << "match case 1.1, closest two points:" << b1 << a4 << endl;
        }
	} else if (b3.x > a2.x && b3.y < a2.y) {
		shortestDist = sqrt(pow(b3.x - a2.x, 2) + pow(b3.y - a2.y, 2));
        if (DEBUG_CLOSE_DISTANCE) {
        	cout << "match case 1.2" << endl;
        }
	} else if (b4.x < a1.x && b4.y < a1.y) {
		shortestDist = sqrt(pow(b4.x - a1.x, 2) + pow(b4.y - a1.y, 2));
        if (DEBUG_CLOSE_DISTANCE) {
        	cout << "match case 1.3" << endl;
        }
	} else if (b2.x < a3.x && b2.y > a3.y) {
		shortestDist = sqrt(pow(b2.x - a3.x, 2) + pow(b2.y - a3.y, 2));
        if (DEBUG_CLOSE_DISTANCE) {
        	cout << "match case 1.4" << endl;
        }
	// case 2:
	} else if (b1.x >= a2.x){
        shortestDist = abs(b1.x - a2.x);
        if (DEBUG_CLOSE_DISTANCE) {
        	cout << "match case 2.1" << endl;
        }
	} else if (b3.y <= a1.y) {
		shortestDist = abs(b3.y - a1.y);
        if (DEBUG_CLOSE_DISTANCE) {
        	cout << "match case 2.2" << endl;
        }
	} else if (b4.x <= a1.x) {
		shortestDist = abs(b4.x - a1.x);
        if (DEBUG_CLOSE_DISTANCE) {
        	cout << "match case 2.3" << endl;
        }
	} else if (b1.y >= a3.y) {
        shortestDist = abs(b1.y - a3.y);
        if (DEBUG_CLOSE_DISTANCE) {
        	cout << "match case 2.4" << endl;
        }
	} else {
		cerr << "WHAT IS THIS CASE?" << a << b << endl;
	}

    if (DEBUG_CLOSE_DISTANCE) {
    	cout << a << " and " << b << " have a closest distance of [" << shortestDist << "]" << endl;
    	// draw rects
//        Mat z(480, 640, CV_8UC3);
//    	rectangle(z, a.tl(), a.br(), Scalar(0, 255, 0), 2, 8, 0);
//    	rectangle(z, b.tl(), b.br(), Scalar(0, 255, 0), 2, 8, 0);
//        imshow("a&b", z);
//        waitKey();
//        z.release();
    }
    if (shortestDist < CLOSE_RECT_THRESHOLD) {
    	return true;
    }


    return false;
}

// use color filter to extract UTDALLAS area in order to
// reduce SURF matching process time.
Mat filter_utdallas_color(Mat a) {
    Mat frameCopy;
    Mat markedImg;
	// convert the color space from RGB to HSV
	cvtColor(a, frameCopy, CV_BGR2HSV); //Change the color format from BGR to HSV
	// color detection of the object.
	inRange(frameCopy, Scalar(lowerH, lowerS, lowerV),
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

	dilate(markedImg, markedImg, Mat(), Point(-1, -1), 10);

	return markedImg;
}

// detect human body boundaries from diff image.
Rect edge_detect_callback(Mat& rawCopy, Mat& diff) {
	Mat detected_edge;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	int thresh = 14;
	int ratio = 3;
	int kernel_size = 3;

	// Detect edges using Canny
	Canny(diff, detected_edge, thresh, thresh * ratio,
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
	for (unsigned int i = 0; i < contours.size(); i++) {
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}

	vector<Rect> boundRectFiltered; //  = boundRect;
	vector<vector<Point> > contoursPolyFiltered; //= contours_poly;
	vector<Rect> boundRectRuleout; //  = boundRect;
	vector<vector<Point> > contoursPolyRuleout; //= contours_poly;

	// remove inner rectangles.
	for (unsigned int i = 0; i < boundRect.size(); i++) {
		Rect r = boundRect[i];
		unsigned int j = 0;
		for (; j < boundRect.size(); j++) {
			if (j != i && boundRect[i] != boundRect[j] && (r & boundRect[j]) == r) {
//				cout << r << " is in " << boundRect[j] << endl;
				break;
			}
		}
		if (j == boundRect.size()) {
			boundRectFiltered.push_back(r);
			contoursPolyFiltered.push_back(contours_poly[i]);
		} else {
            // for DEBUG use.
			boundRectRuleout.push_back(r);
			contoursPolyRuleout.push_back(contours_poly[i]);
		}
	}

	vector<Rect> boundRectMerged; //  = boundRect;
	vector<vector<Point> > contoursPolyMerged; //= contours_poly;

	// merge overlapped contours. 5 iterations to make sure they are correctly merged.
    for (int iter = 0; iter < 5; iter++) {
        boundRectMerged.clear();
		for (unsigned int i = 0; i < boundRectFiltered.size(); i++) {
			Rect r = boundRectFiltered[i];
			Rect container(r);
			unsigned int j = i + 1;
			while (j < boundRectFiltered.size()) {
				Rect s = boundRectFiltered[j];
//				Point centerS(s.x + s.width / 2, s.y + s.height / 2);
//				Rect insection = container & s;
//			if (insection.area() != 0
//					|| sqrt(pow(centerS.x - centerR.x, 2) + pow(centerS.y - centerR.y, 2)) < 500) {
				if (isCloseEnough(r, s)) {
					container |= s;
					boundRectFiltered.erase(boundRectFiltered.begin() + j);
				} else {
					j++;
				}

//			Point centerS = (s.tl() + s.br()) / 2;
//			double centerDistance = sqrt(pow(centerR.x - centerS.x, 2) + pow(centerR.y - centerS.y, 2));
//			double rectDistance = min()
			}

			boundRectMerged.push_back(container);
		}
        boundRectFiltered = boundRectMerged;
    }

//	Rect container;
//	for (int i = 0; i < boundRectFiltered.size(); i++) {
//		Rect r = boundRectFiltered[i];
//		if (0 == container.area()) {
//			container = Rect(r);
//		} else {
//			container |= r;
//		}
//	}
//	boundRectMerged.push_back(container);

	if (DEBUG) {
		cout << "---" << endl;
		for (unsigned int i = 0; i < boundRectMerged.size(); i++) {
			cout << "Rect group " << i + 1 << ":" << boundRectMerged[i] << endl;
		}
		cout << "---" << endl;
	}

//  DEBUG code.
//	// Draw polygonal contour + bounding rects
//	for (int i = 0; i < boundRectFiltered.size(); i++) {
//		// the yellow contour
//		drawContours(rawCopy, contoursPolyFiltered, i, Scalar(0, 255, 255), 1, 8, vector<Vec4i>(), 0,
//				Point());
//	}

//	for (int i = 0; i < boundRectMerged.size(); i++) {
		// draw bounding rect
//		rectangle(rawCopy, boundRectFiltered[i].tl(), boundRectFiltered[i].br(), Scalar(0, 255, 255), 1, 8,
//				0);
//	}

//	for (int i = 0; i < boundRectMerged.size(); i++) {
		// draw bounding rect
//		rectangle(rawCopy, boundRectMerged[i].tl(), boundRectMerged[i].br(), Scalar(0, 0, 255), 2, 8,
//				0);
//	}

//	for (int i = 0; false && i < boundRectRuleout.size(); i++) {
		// the yellow contour
//		drawContours(rawCopy, contoursPolyRuleout, i, Scalar(0, 255, 255), 1, 8, vector<Vec4i>(), 0,
//				Point());
		// draw bounding rect
//		rectangle(rawCopy, boundRectRuleout[i].tl(), boundRectRuleout[i].br(), Scalar(0, 255, 0), 2, 8,
//				0);
//	}

    // select the right person
    int rightPersonIndex = -1;
    for (unsigned int i = 0; i < boundRectMerged.size(); i++) {
        int count = 0;
    	for (unsigned int j = 0; j < SURFMatchedPolygon.size(); j++) {
    		if (boundRectMerged[i].contains(SURFMatchedPolygon[j])) {
    			count++;
    		}
    	}

    	if (count > 2) {
    		// this rectangle contains UTDALLAS
    		// select this one.
            rightPersonIndex = i;
            break;
    	}
    }

    // there is no match from SURF.
    if (-1 == rightPersonIndex) {
    	return Rect(0, 0, 0, 0);
    }

    // draw bounding rect
	rectangle(rawCopy, boundRectMerged[rightPersonIndex].tl(), boundRectMerged[rightPersonIndex].br(),
				Scalar(0, 0, 255), 2, 8, 0);

//    Deprecated method
//    int maxArea = INT32_MIN;
//    int maxIndex = 0;
//	for (int i = 0; i < boundRectMerged.size(); i++) {
//		if (boundRectMerged[i].area() > maxArea) {
//			maxArea = boundRectMerged[i].area();
//            maxIndex = i;
//		}
//	}

	return boundRectMerged[rightPersonIndex];
}

void trackMovement(Rect keyPersonRect) {
	static vector<Point> personMovePath;
    Mat output = Mat::zeros(480, 640, CV_8UC3);
    personMovePath.push_back(Point(keyPersonRect.x + keyPersonRect.width / 2,
    		keyPersonRect.y + keyPersonRect.height / 2));
    // if overflows, remove the oldest one.
    if (personMovePath.size() > 150) {
    	personMovePath.erase(personMovePath.begin());
    }
    for (unsigned int i = 0; i < personMovePath.size() - 1; i++) {
        Scalar color(0, (int) (255 * (1 - (float) i / personMovePath.size())), (int) (255 * ((float) i / personMovePath.size())));
    	line(output, personMovePath[i], personMovePath[i + 1], color, 1, CV_AA);
    }
    imshow("Movement", output);
}

int main(int argc, char** argv) {
//    for DEBUG purpose
//	VideoCapture capture("2014-04-12 21.22.05.mp4");
//	VideoCapture capture("2014-04-12 21.23.42.mp4");
//	VideoCapture capture("2014-05-04-115055.webm");
//	VideoCapture capture("2014-05-04-115210.webm");
//	VideoCapture capture("2014-05-07-103938.webm");
//	VideoCapture capture("2014-05-07-104009.webm");
//	VideoCapture capture("2014-05-11-130539.webm");
//	VideoCapture capture("2014-05-11-130648.webm");
//	VideoCapture capture(0);
//	Mat bgImg = imread("2014-05-07-103917.jpg");
//	Mat bgImg = imread("2014-05-04-115048.jpg");
//	Mat bgImg = imread("2014-05-11-130531.jpg");
//	VideoCapture capture("2014-04-12 21.20.28.mp4");
//	frameTemplate = imread("2014-04-12 21.24.02_300x128.jpg");

	VideoCapture capture(0);
    if (!capture.isOpened()) {
    	cerr << "WebCam cannot be opened!" << endl;
    }

    namedWindow("raw");
    setMouseCallback("raw", onClick, NULL);
    Mat bgImg;
    int c = -1;
    Mat frame;
	for (;;) {
		capture >> frame;
        c++;
		if (frame.empty()) {
			return 1;
		}
        if (0 == c) {
        	// first frame as background.
            bgImg = frame.clone();
        	cvtColor(bgImg, bgImg, CV_BGR2GRAY);
        	GaussianBlur(bgImg, bgImg, Size(3, 3), 0, 0);
            continue;
        }
		Mat frameCopy = frame.clone();
		Mat rawCopyForSelection = frame.clone();
        // background diff method works on grayscale images.
		cvtColor(frameCopy, frameCopy, CV_BGR2GRAY);

		Mat diff;
        // remove noises
		GaussianBlur(diff, diff, Size(3, 3), 0, 0);
        // foreground - background = diff.
		absdiff(frameCopy, bgImg, diff);
//      //for debug purpose
//		double min;
//		double max;
//		Point minLoc;
//		Point maxLoc;
//		minMaxLoc(diff, &min, &max, &minLoc, &maxLoc);
        //
        // only take obvious differences in account.
		inRange(diff, 50, 250, diff);
        // remove holes.
		erode(diff, diff, Mat(), Point(-1, -1), 2);
		dilate(diff, diff, Mat(), Point(-1, -1), 10);
		erode(diff, diff, Mat(), Point(-1, -1), 2);

        // foreground == background.
		// we skip processing this frame.
        if (countNonZero(diff) == 0) {
    		imshow("raw", frame.clone());
            int k = waitKey(20);
            if (k == 'p' || k == 'Q') {
            	return 0;
            }
        	continue;
        }
		rawCopyForDraw = frame.clone();
        if (startRecognition) {
			Mat rawCopyForSURF = frame.clone();
			temocTracking(rawCopyForSURF);
    		keyPersonRect = edge_detect_callback(rawCopyForDraw, diff);
    		if (keyPersonRect.area() != 0) {
                trackMovement(keyPersonRect);
    		}
        }

        // show the [foreground - background] diff binary image.
		imshow("diff", diff);
		imshow("raw", rawCopyForDraw);

        // loop until we are not paused.
        while (true) {
        	int key = waitKey(20);
            if (key == 'p' || key == 'P') {
                if (paused) {
                    // paused -> non-paused.
                	// we update the SURF template
                    if (selection.width > 0 && selection.height > 0) {
                    	frameTemplate = roi.clone();
                        // after we copied ROI, clear selection.
                        selection = Rect(0, 0, 0, 0);
                    }
                    startRecognition = true;
                }
                // toggle the pause state.
            	paused = !paused;
            } else if (key == 'q' || key == 'Q') {
                // quit
            	return 0;
            }
            // update ROI in a different window
            // if valid.
            roi = rawCopyForSelection;
    		if (selection.width > 0 && selection.height > 0) {
    			roi = rawCopyForSelection.clone()(selection);
                if (DEBUG) {
                	cout << "selection: " << selection << endl;
                }
    		}
    		imshow("sel", roi);
            // not paused, we move onto next frame.
            if (!paused) {
            	break;
            }
        }
	}
	return 0;
}
