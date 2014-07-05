#include "codebook.h"

int CVCONTOUR_APPROX_LEVEL = 2;
int CVCLOSE_ITR = 1;

#define CV_CVX_WHITE	CV_RGB(0xff,0xff,0xff)
#define CV_CVX_BLACK	CV_RGB(0x00,0x00,0x00)

int cvupdateCodeBook(uchar *p, codeBook &c, unsigned *cbBounds,
		int numChannels) {

	if (c.numEntries == 0)
		c.t = 0;
	c.t += 1;
	int n;
	unsigned int high[3], low[3];
	for (n = 0; n < numChannels; n++) {
		high[n] = *(p + n) + *(cbBounds + n);
		if (high[n] > 255)
			high[n] = 255;
		low[n] = *(p + n) - *(cbBounds + n);
		if (low[n] < 0)
			low[n] = 0;
	}
	int matchChannel;
	int i = 0;
	for (; i < c.numEntries; i++) {
		matchChannel = 0;
		for (n = 0; n < numChannels; n++) {
			if ((c.cb[i]->learnLow[n] <= *(p + n))
					&& (*(p + n) <= c.cb[i]->learnHigh[n])) {
				matchChannel++;
			}
		}
		if (matchChannel == numChannels) {
			c.cb[i]->t_last_update = c.t;
			for (n = 0; n < numChannels; n++) {
				if (c.cb[i]->max[n] < *(p + n)) {
					c.cb[i]->max[n] = *(p + n);
				} else if (c.cb[i]->min[n] > *(p + n)) {
					c.cb[i]->min[n] = *(p + n);
				}
			}
			break;
		}
	}

	for (int s = 0; s < c.numEntries; s++) {
		int negRun = c.t - c.cb[s]->t_last_update;
		if (c.cb[s]->stale < negRun)
			c.cb[s]->stale = negRun;
	}

	if (i == c.numEntries) {
		code_element **foo = new code_element*[c.numEntries + 1];
		for (int ii = 0; ii < c.numEntries; ii++) {
			foo[ii] = c.cb[ii];
		}
		foo[c.numEntries] = new code_element;
		if (c.numEntries)
			delete[] c.cb;
		c.cb = foo;
		for (n = 0; n < numChannels; n++) {
			c.cb[c.numEntries]->learnHigh[n] = high[n];
			c.cb[c.numEntries]->learnLow[n] = low[n];
			c.cb[c.numEntries]->max[n] = *(p + n);
			c.cb[c.numEntries]->min[n] = *(p + n);
		}
		c.cb[c.numEntries]->t_last_update = c.t;
		c.cb[c.numEntries]->stale = 0;
		c.numEntries += 1;
	}

	for (n = 0; n < numChannels; n++) {
		if (c.cb[i]->learnHigh[n] < high[n])
			c.cb[i]->learnHigh[n] += 1;
		if (c.cb[i]->learnLow[n] > low[n])
			c.cb[i]->learnLow[n] -= 1;
	}

	return (i);
}

uchar cvbackgroundDiff(uchar *p, codeBook &c, int numChannels, int *minMod,
		int *maxMod) {
	int matchChannel;
	int i = 0;
	for (; i < c.numEntries; i++) {
		matchChannel = 0;
		for (int n = 0; n < numChannels; n++) {
			if ((c.cb[i]->min[n] - minMod[n] <= *(p + n))
					&& (*(p + n) <= c.cb[i]->max[n] + maxMod[n])) {
				matchChannel++;
			} else {
				break;
			}
		}
		if (matchChannel == numChannels) {
			break;
		}
	}
	if (i >= c.numEntries)
		return (255);
	return (0);
}

int cvclearStaleEntries(codeBook &c) {
	int staleThresh = c.t >> 1;
	int *keep = new int[c.numEntries];
	int keepCnt = 0;
	for (int i = 0; i < c.numEntries; i++) {
		if (c.cb[i]->stale > staleThresh)
			keep[i] = 0;
		else {
			keep[i] = 1;
			keepCnt += 1;
		}
	}
	c.t = 0;
	code_element **foo = new code_element*[keepCnt];
	int k = 0;
	for (int ii = 0; ii < c.numEntries; ii++) {
		if (keep[ii]) {
			foo[k] = c.cb[ii];
			foo[k]->stale = 0;
			foo[k]->t_last_update = 0;
			k++;
		}
	}
	delete[] keep;
	delete[] c.cb;
	c.cb = foo;
	int numCleared = c.numEntries - keepCnt;
	c.numEntries = keepCnt;
	return (numCleared);
}

int cvcountSegmentation(codeBook *c, IplImage *I, int numChannels, int *minMod,
		int *maxMod) {
	int count = 0, i;
	uchar *pColor;
	int imageLen = I->width * I->height;

	pColor = (uchar *) ((I)->imageData);
	for (i = 0; i < imageLen; i++) {
		if (cvbackgroundDiff(pColor, c[i], numChannels, minMod, maxMod))
			count++;
		pColor += 3;
	}
	return (count);
}

void cvconnectedComponents(IplImage *mask, int poly1_hull0, float perimScale,
		int *num, CvRect *bbs, CvPoint *centers) {
	static CvMemStorage* mem_storage = NULL;
	static CvSeq* contours = NULL;
	cvMorphologyEx(mask, mask, NULL, NULL, CV_MOP_OPEN, CVCLOSE_ITR);
	cvMorphologyEx(mask, mask, NULL, NULL, CV_MOP_CLOSE, CVCLOSE_ITR);

	if (mem_storage == NULL)
		mem_storage = cvCreateMemStorage(0);
	else
		cvClearMemStorage(mem_storage);

	CvContourScanner scanner = cvStartFindContours(mask, mem_storage,
			sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	CvSeq* c;
	int numCont = 0;
	while ((c = cvFindNextContour(scanner)) != NULL) {
		double len = cvContourPerimeter(c);
		double q = (mask->height + mask->width) / perimScale; //calculate perimeter len threshold
		if (len < q) {
			cvSubstituteContour(scanner, NULL);
		} else {
			CvSeq* c_new;
			if (poly1_hull0)
				c_new = cvApproxPoly(c, sizeof(CvContour), mem_storage,
						CV_POLY_APPROX_DP, CVCONTOUR_APPROX_LEVEL, 0);
			else
				c_new = cvConvexHull2(c, mem_storage, CV_CLOCKWISE, 1);
			cvSubstituteContour(scanner, c_new);
			numCont++;
		}
	}
	contours = cvEndFindContours(&scanner);

	cvZero(mask);
	IplImage *maskTemp;
	if (num != NULL) {
		int N = *num, numFilled = 0, i = 0;
		CvMoments moments;
		double M00, M01, M10;
		maskTemp = cvCloneImage(mask);
		for (i = 0, c = contours; c != NULL; c = c->h_next, i++) {
			if (i < N) {
				cvDrawContours(maskTemp, c, CV_CVX_WHITE, CV_CVX_WHITE, -1,
						CV_FILLED, 8);
				if (centers != NULL) {
					cvMoments(maskTemp, &moments, 1);
					M00 = cvGetSpatialMoment(&moments, 0, 0);
					M10 = cvGetSpatialMoment(&moments, 1, 0);
					M01 = cvGetSpatialMoment(&moments, 0, 1);
					centers[i].x = (int) (M10 / M00);
					centers[i].y = (int) (M01 / M00);
				}
				if (bbs != NULL) {
					bbs[i] = cvBoundingRect(c);
				}
				cvZero(maskTemp);
				numFilled++;
			}
			cvDrawContours(mask, c, CV_CVX_WHITE, CV_CVX_WHITE, -1, CV_FILLED,
					8); //draw to central mask
		}
		*num = numFilled;
		cvReleaseImage(&maskTemp);
	} else {
		for (c = contours; c != NULL; c = c->h_next) {
			cvDrawContours(mask, c, CV_CVX_WHITE, CV_CVX_BLACK, -1, CV_FILLED,
					8);
		}
	}
}

