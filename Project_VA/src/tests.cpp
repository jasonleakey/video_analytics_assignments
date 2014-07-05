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

int main(int argc, char **argv) {
//	Rect a(0, 0, 3, 3);
//	Rect b(0, 2, 3, 3);
//	Rect c(0, 9, 3, 3);
//	Rect d(0, 0, 1, 1);
//	Rect z = a & b;
//	Rect y = c & b;
//	Rect x = c | d;
//	cout << z << endl;
//	cout << y << endl;
//	cout << x << endl;
    Mat canvas(480, 640, CV_8U);
    Rect a(175, 158, 108, 306);
    Rect b(283, 39, 338, 440);
    rectangle(canvas, a.tl(), a.br(), Scalar(255, 0, 0), 1, 8, 0);
    rectangle(canvas, b.tl(), b.br(), Scalar(255, 0, 0), 1, 8, 0);
    imshow("rect", canvas);
    waitKey(0);
    return 0;
}
