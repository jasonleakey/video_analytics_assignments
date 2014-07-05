/*
 * SIFT.cpp
 *
 *  Created on: Mar 22, 2014
 *      Author: jasonleakey
 */

// sift_test.cpp : 定义控制台应用程序的入口点。
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"//因为在属性中已经配置了opencv等目录，所以把其当成了本地目录一样
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

void readme();

int main(int argc, char* argv[]) {
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cerr << "open camera failed" << endl;
		return -1;
	}

//	Mat img_1 = imread("./data/train/188.temoc/01.jpg",
//			CV_LOAD_IMAGE_GRAYSCALE); //宏定义时CV_LOAD_IMAGE_GRAYSCALE=0，也就是读取灰度图像
	while (true) {
//		Mat img_2 = imread("./data/train/188.temoc/13.jpg",

//				CV_LOAD_IMAGE_GRAYSCALE); //一定要记得这里路径的斜线方向，这与Matlab里面是相反的
		Mat frame;
		cap >> frame;
		if (frame.empty()) {
			cerr << "could not get frame from camera" << endl;
			continue;
		}
		Mat img_2;
		cvtColor(frame, img_2, CV_RGB2GRAY);
//	Mat img_1 = imread("box.png", CV_LOAD_IMAGE_GRAYSCALE); //宏定义时CV_LOAD_IMAGE_GRAYSCALE=0，也就是读取灰度图像
//	Mat img_2 = imread("box_in_scene.png", CV_LOAD_IMAGE_GRAYSCALE); //一定要记得这里路径的斜线方向，这与Matlab里面是相反的

		if (!img_1.data || !img_2.data) //如果数据为空
				{
			cout << "opencv error" << endl;
			return -1;
		}
		cout << "open right" << endl;

		//第一步，用SIFT算子检测关键点
		int minHessian = 400;

//	SiftFeatureDetector detector; //构造函数采用内部默认的
		SurfFeatureDetector detector(minHessian); //构造函数采用内部默认的
		std::vector<KeyPoint> keypoints_1, keypoints_2; //构造2个专门由点组成的点向量用来存储特征点

		detector.detect(img_1, keypoints_1); //将img_1图像中检测到的特征点存储起来放在keypoints_1中
		detector.detect(img_2, keypoints_2); //同理

		//在图像中画出特征点
		Mat img_keypoints_1, img_keypoints_2;

		drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1),
				DrawMatchesFlags::DEFAULT); //在内存中画出特征点
		drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1),
				DrawMatchesFlags::DEFAULT);

		imshow("sift_keypoints_1", img_keypoints_1); //显示特征点
		imshow("sift_keypoints_2", img_keypoints_2);

		//计算特征向量
//		SiftDescriptorExtractor extractor; //定义描述子对象
		SurfDescriptorExtractor extractor; //定义描述子对象

		Mat descriptors_1, descriptors_2; //存放特征向量的矩阵

		extractor.compute(img_1, keypoints_1, descriptors_1); //计算特征向量
		extractor.compute(img_2, keypoints_2, descriptors_2);

		double max_dist = 0;
		double min_dist = 100;

		//用burte force进行匹配特征向量
//	BFMatcher matcher(NORM_L2); //定义一个burte force matcher对象
		FlannBasedMatcher matcher; //定义一个burte force matcher对象
		vector<DMatch> matches;
		matcher.match(descriptors_1, descriptors_2, matches);

		//-- Quick calculation of max and min distances between keypoints
		for (int i = 0; i < descriptors_1.rows; i++) {
			double dist = matches[i].distance;
			if (dist < min_dist)
				min_dist = dist;
			if (dist > max_dist)
				max_dist = dist;
		}

		cout << "-- Max dist : %f \n" << max_dist;
		cout << "-- Min dist : %f \n" << min_dist;

		//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
		vector<DMatch> good_matches;

		for (int i = 0; i < descriptors_1.rows; i++) {
			if (matches[i].distance < 3 * min_dist) {
				good_matches.push_back(matches[i]);
			}
		}

		//绘制匹配线段
		Mat img_matches;
//	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches); //将匹配出来的结果放入内存img_matches中

		drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches,
				img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(),
				DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		//-- Localize the object
		std::vector<Point2f> obj;
		std::vector<Point2f> scene;

		for (int i = 0; i < good_matches.size(); i++) {
			//-- Get the keypoints from the good matches
			obj.push_back(keypoints_1[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints_2[good_matches[i].trainIdx].pt);
		}

		Mat H = findHomography(obj, scene, CV_RANSAC);

		//-- Get the corners from the image_1 ( the object to be "detected" )
		std::vector<Point2f> obj_corners(4);
		obj_corners[0] = cvPoint(0, 0);
		obj_corners[1] = cvPoint(img_1.cols, 0);
		obj_corners[2] = cvPoint(img_1.cols, img_1.rows);
		obj_corners[3] = cvPoint(0, img_1.rows);
		std::vector<Point2f> scene_corners(4);

		perspectiveTransform(obj_corners, scene_corners, H);

		//-- Draw lines between the corners (the mapped object in the scene - image_2 )
		line(img_matches, scene_corners[0] + Point2f(img_1.cols, 0),
				scene_corners[1] + Point2f(img_1.cols, 0), Scalar(0, 255, 0),
				4);
		line(img_matches, scene_corners[1] + Point2f(img_1.cols, 0),
				scene_corners[2] + Point2f(img_1.cols, 0), Scalar(0, 255, 0),
				4);
		line(img_matches, scene_corners[2] + Point2f(img_1.cols, 0),
				scene_corners[3] + Point2f(img_1.cols, 0), Scalar(0, 255, 0),
				4);
		line(img_matches, scene_corners[3] + Point2f(img_1.cols, 0),
				scene_corners[0] + Point2f(img_1.cols, 0), Scalar(0, 255, 0),
				4);

		//显示匹配线段
		imshow("sift_Matches", img_matches); //显示的标题为Matches
		int key = waitKey(40);
		// Exit if Keypress 'q' or 'Esc'
		if (key == 'q' || key == 27) {
			break;
		}
	}
	return 0;
}
