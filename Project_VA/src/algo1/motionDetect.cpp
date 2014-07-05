#include <time.h>
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cv.h"
#include "highgui.h"
#include "background.h"
#include "codebook.h"

codeBook *cB;

const int mtN = 3;
const double MHI_DUR = 0.5;
const int CONTOUR_MAX_AERA = 500;

int mtlast = 0;
int motionNum = 0;
IplImage *mhi = 0;
IplImage **mtbuf = 0;

const int T1 = 30;
const int T2 = -30;

bool ch[CHANNELS];
int maxMod[CHANNELS];
int minMod[CHANNELS];
unsigned cbBounds[CHANNELS];

int imageLen = 0;
int nChannels = CHANNELS;
uchar *pColor;

void lumenComp(IplImage *pFrame) {
	int histogram[256];
	int colorr, colorg, colorb;

	const float thresholdco = 0.05;

	for (int i = 0; i < 256; i++)
		histogram[i] = 0;

	IplImage *pFrame_gray = cvCreateImage(cvSize(pFrame->width, pFrame->height),
			8, 1);
	cvCvtColor(pFrame, pFrame_gray, CV_BGR2GRAY);

	cvSmooth(pFrame, pFrame, CV_GAUSSIAN, 3, 0, 0);

	for (int i = 0; i < pFrame_gray->height; i++)
		for (int j = 0; j < pFrame_gray->width; j++) {
			colorb = ((uchar*) (pFrame->imageData + pFrame->widthStep * i))[j
					* 3];
			colorg = ((uchar*) (pFrame->imageData + pFrame->widthStep * i))[j
					* 3 + 1];
			colorr = ((uchar*) (pFrame->imageData + pFrame->widthStep * i))[j
					* 3 + 2];
			int gray = ((uchar*) (pFrame_gray->imageData
					+ i * pFrame_gray->widthStep))[j];
			histogram[gray]++;
		}

	int calnum = 0;
	int total = pFrame->width * pFrame->height;
	int num;

	for (int i = 0; i < 256; i++) {
		if ((float) calnum / total < thresholdco) {
			calnum += histogram[255 - i];
			num = i;
		} else
			break;
	}

	int averagegray = 0;
	calnum = 0;

	for (int i = 255; i >= 255 - num; i--) {
		averagegray += histogram[i] * i;
		calnum += histogram[i];
	}
	averagegray /= calnum;
	float co = 255.0 / (float) averagegray;

	for (int i = 0; i < pFrame->height; i++)
		for (int j = 0; j < pFrame->width; j++) {
			colorb = ((uchar*) (pFrame->imageData + pFrame->widthStep * i))[j
					* 3];
			colorb *= co;
			if (colorb > 255)
				colorb = 255;
			((uchar*) (pFrame->imageData + pFrame->widthStep * i))[j * 3] =
					colorb;
			colorb = ((uchar*) (pFrame->imageData + pFrame->widthStep * i))[j
					* 3 + 1];
			colorb *= co;
			if (colorb > 255)
				colorb = 255;
			((uchar*) (pFrame->imageData + pFrame->widthStep * i))[j * 3 + 1] =
					colorb;
			colorb = ((uchar*) (pFrame->imageData + pFrame->widthStep * i))[j
					* 3 + 2];
			colorb *= co;
			if (colorb > 255)
				colorb = 255;
			((uchar*) (pFrame->imageData + pFrame->widthStep * i))[j * 3 + 2] =
					colorb;
		}
}

void update_mhi(IplImage* mtsilh, IplImage* ctimg, IplImage* mtdst,
		int diff_threshold) {
	double timestamp = clock() / 100.;
	CvSize mtsize = cvSize(mtsilh->width, mtsilh->height);
	int mti;
	IplImage* mtpyr = cvCreateImage(
			cvSize((mtsize.width & -2) / 2, (mtsize.height & -2) / 2), 8, 1);
	CvMemStorage *mtstor;
	CvSeq *mtcont;

	if (!mhi || mhi->width != mtsize.width || mhi->height != mtsize.height) {
		if (mtbuf == 0) {
			mtbuf = (IplImage**) malloc(mtN * sizeof(mtbuf[0]));
			memset(mtbuf, 0, mtN * sizeof(mtbuf[0]));
		}

		for (mti = 0; mti < mtN; mti++) {
			cvReleaseImage(&mtbuf[mti]);
			mtbuf[mti] = cvCreateImage(mtsize, IPL_DEPTH_8U, 1);
			cvZero(mtbuf[mti]);
		}

		cvReleaseImage(&mhi);
		mhi = cvCreateImage(mtsize, IPL_DEPTH_32F, 1);
		cvZero(mhi);
	}

	cvThreshold(mtsilh, mtsilh, 30, 255, CV_THRESH_BINARY);
	cvUpdateMotionHistory(mtsilh, mhi, timestamp, MHI_DUR);
	cvCvtScale(mhi, mtdst, 255. / MHI_DUR,
			(MHI_DUR - timestamp) * 255. / MHI_DUR);
	cvCvtScale(mhi, mtdst, 255. / MHI_DUR, 0);

	cvSmooth(mtdst, mtdst, CV_MEDIAN, 3, 0, 0, 0);

	cvPyrDown(mtdst, mtpyr, 7);
	cvDilate(mtpyr, mtpyr, 0, 1);
	cvPyrUp(mtpyr, mtdst, 7);

	mtstor = cvCreateMemStorage(0);
	mtcont = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint),
			mtstor);

	cvFindContours(mtdst, mtstor, &mtcont, sizeof(CvContour), CV_RETR_LIST,
			CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
	for (; mtcont; mtcont = mtcont->h_next) {

		CvRect mtr = ((CvContour*) mtcont)->rect;
		if ((mtr.height * mtr.width > CONTOUR_MAX_AERA)
				&& (mtr.height > mtr.width)) {
			cvRectangle(ctimg, cvPoint(mtr.x, mtr.y),
					cvPoint(mtr.x + mtr.width, mtr.y + mtr.height),
					CV_RGB(0, 0, 255), 1, CV_AA, 0);
			cvDrawContours(ctimg, mtcont, CV_RGB(255, 0, 0), CV_RGB(0, 0, 255),
					0, 1, 8, cvPoint(0, 0));
			motionNum++;
		}
	}
	cvReleaseMemStorage(&mtstor);
	cvReleaseImage(&mtpyr);
}

int main(int argc, char** argv) {
	CvCapture* capture = 0;

	IplImage* mtmotion = 0;
	IplImage* showImage = 0;
	IplImage* rawImage = 0, *yuvImage = 0;
	IplImage *ImaskAVG = 0, *ImaskAVGCC = 0;
	IplImage *ImaskCodeBook = 0, *ImaskCodeBookCC = 0;

	int sdx, sdy, sdt;

	IplImage* sdFrImg = NULL;
	IplImage* sdBkImg = NULL;

	CvMat* rawImageMat = NULL;
	CvMat* sdFrMat = NULL;
	CvMat* sdBkMat = NULL;

	int c;
	int startcapture = 1;
	int endcapture = 30;

	maxMod[0] = 3;
	minMod[0] = 10;
	maxMod[1] = 1;
	minMod[1] = 1;
	maxMod[2] = 1;
	minMod[2] = 1;

	float scalehigh = HIGH_SCALE_NUM;
	float scalelow = LOW_SCALE_NUM;

//	capture = cvCaptureFromCAM(0);
	char* filename="2014-05-04-115055.webm";

	capture = cvCreateFileCapture(filename);

	if (capture) {
		cvNamedWindow("Carema", 1);
		cvNamedWindow("Foreground", 1);
		cvNamedWindow("Background", 1);
		cvMoveWindow("Carema", 0, 0);
		cvMoveWindow("Foreground", 640, 0);
		cvMoveWindow("Background", 0, 640);

		int i = -1;

		for (;;) {
			rawImage = cvQueryFrame(capture);
			++i;

			if (!rawImage)
				break;

			if (i == 0) {
				printf("\n . . .Please wait for it . . .\n");
				AllocateImages(rawImage);

				scaleHigh(scalehigh);
				scaleLow(scalelow);

				sdBkImg = cvCreateImage(
						cvSize(rawImage->width, rawImage->height), IPL_DEPTH_8U,
						1);
				sdFrImg = cvCreateImage(
						cvSize(rawImage->width, rawImage->height), IPL_DEPTH_8U,
						1);
				sdBkMat = cvCreateMat(rawImage->height, rawImage->width,
						CV_32FC1);
				sdFrMat = cvCreateMat(rawImage->height, rawImage->width,
						CV_32FC1);

				rawImageMat = cvCreateMat(rawImage->height, rawImage->width,
						CV_32FC1);

				sdBkImg->origin = 1;
				sdFrImg->origin = 1;

				cvCvtColor(rawImage, sdBkImg, CV_BGR2GRAY);
				cvCvtColor(rawImage, sdFrImg, CV_BGR2GRAY);

				cvConvert(sdFrImg, rawImageMat);
				cvConvert(sdFrImg, sdFrMat);
				cvConvert(sdFrImg, sdBkMat);

				ImaskAVG = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 1);
				ImaskAVGCC = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U,
						1);
				cvSet(ImaskAVG, cvScalar(255));

				yuvImage = cvCloneImage(rawImage);

				ImaskCodeBook = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U,
						1);
				ImaskCodeBookCC = cvCreateImage(cvGetSize(rawImage),
						IPL_DEPTH_8U, 1);
				cvSet(ImaskCodeBook, cvScalar(255));

				imageLen = rawImage->width * rawImage->height;

				cB = new codeBook[imageLen];

				for (int f = 0; f < imageLen; f++) {
					cB[f].numEntries = 0;
				}

				for (int nc = 0; nc < nChannels; nc++) {
					cbBounds[nc] = 10;
				}

				ch[0] = true;
				ch[1] = true;
				ch[2] = true;
			}

			if (rawImage) {
				lumenComp(rawImage);

				showImage = cvCreateImage(
						cvSize(rawImage->width, rawImage->height),
						rawImage->depth, rawImage->nChannels);
				cvCopy(rawImage, showImage, NULL);
//				cvFlip(showImage, showImage, 0);

				cvCvtColor(rawImage, sdFrImg, CV_BGR2GRAY);
				cvCvtColor(rawImage, yuvImage, CV_BGR2YCrCb);
				cvConvert(sdFrImg, rawImageMat);

				if (i >= startcapture && i < endcapture) {
					accumulateBackground(rawImage);
					pColor = (uchar *) ((yuvImage)->imageData);

					for (int c = 0; c < imageLen; c++) {
						cvupdateCodeBook(pColor, cB[c], cbBounds, nChannels);
						pColor += 3;
					}
				}

				if (i == endcapture) {
					createModelsfromStats();
				}

				if (i >= endcapture) {
					backgroundDiff(rawImage, ImaskAVG);
					cvCopy(ImaskAVG, ImaskAVGCC);
					cvconnectedComponents(ImaskAVGCC);

					uchar maskPixelCodeBook;
					pColor = (uchar *) ((yuvImage)->imageData);
					uchar *pMask = (uchar *) ((ImaskCodeBook)->imageData);

					for (int c = 0; c < imageLen; c++) {
						maskPixelCodeBook = cvbackgroundDiff(pColor, cB[c],
								nChannels, minMod, maxMod);
						*pMask++ = maskPixelCodeBook;
						pColor += 3;
					}

					cvCopy(ImaskCodeBook, ImaskCodeBookCC);
					for (sdy = 0; sdy < rawImage->height; sdy++)
						for (sdx = 0; sdx < rawImage->width; sdx++) {
							sdt = ((uchar*) (sdFrImg->imageData
									+ sdFrImg->widthStep * sdy))[sdx]
									- ((uchar*) (sdBkImg->imageData
											+ sdBkImg->widthStep * sdy))[sdx];
							if (sdt > T1 || sdt < T2)
								((uchar*) (ImaskCodeBookCC->imageData
										+ ImaskCodeBookCC->widthStep * sdy))[sdx] =
										255;
							else
								((uchar*) (ImaskCodeBookCC->imageData
										+ ImaskCodeBookCC->widthStep * sdy))[sdx] =
										0;
						}
					cvconnectedComponents(ImaskCodeBookCC);
				}

				if (!mtmotion) {
					mtmotion = cvCreateImage(
							cvSize(rawImage->width, rawImage->height), 8, 1);
					cvZero(mtmotion);
					mtmotion->origin = rawImage->origin;
				}

//				cvFlip(ImaskCodeBookCC, ImaskCodeBookCC, 0);
				cvShowImage("Foreground", ImaskCodeBookCC);

				update_mhi(ImaskCodeBookCC, showImage, mtmotion, 60);

				printf("Number of moving objects: %d\n", motionNum);
				motionNum = 0;

				cvShowImage("Carema", showImage);

				c = cvWaitKey(10) & 0xFF;
				if (c == 27)
					break;
			}
		}

		cvReleaseCapture(&capture);
		cvDestroyWindow("Carema");
		cvDestroyWindow("Foreground");

		DeallocateImages();
		cvReleaseImage(&mtmotion);

		cvReleaseImage(&sdFrImg);
		cvReleaseImage(&sdBkImg);
		cvReleaseImage(&showImage);

		cvReleaseMat(&sdFrMat);
		cvReleaseMat(&sdBkMat);
		cvReleaseMat(&rawImageMat);

		if (yuvImage)
			cvReleaseImage(&yuvImage);
		if (ImaskAVG)
			cvReleaseImage(&ImaskAVG);
		if (ImaskAVGCC)
			cvReleaseImage(&ImaskAVGCC);
		if (ImaskCodeBook)
			cvReleaseImage(&ImaskCodeBook);
		if (ImaskCodeBookCC)
			cvReleaseImage(&ImaskCodeBookCC);

		delete[] cB;
	} else {
		printf("\n\nWarning, Something wrong with the parameters\n\n");
	}
	return 0;
}

