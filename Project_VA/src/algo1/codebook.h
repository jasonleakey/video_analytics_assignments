#ifndef CVYUV_CB
#define CVYUV_CB

#include <cv.h>
#include <highgui.h>
#include <cxcore.h>

#define CHANNELS 3

typedef struct ce {
	uchar learnHigh[CHANNELS];
	uchar learnLow[CHANNELS];
	uchar max[CHANNELS];
	uchar min[CHANNELS];
	int t_last_update;
	int stale;
} code_element;

typedef struct code_book {
	code_element **cb;
	int numEntries;
	int t;
} codeBook;

int cvupdateCodeBook(uchar *p, codeBook &c, unsigned *cbBounds,
		int numChannels = 3);

uchar cvbackgroundDiff(uchar *p, codeBook &c, int numChannels, int *minMod,
		int *maxMod);

int cvclearStaleEntries(codeBook &c);

// eugmentaion(codeBook *c, IplImage *I, int numChannels, int *minMod, int *maxMod);

void cvconnectedComponents(IplImage *mask, int poly1_hull0 = 1,
		float perimScale = 4.0, int *num = NULL, CvRect *bbs = NULL,
		CvPoint *centers = NULL);

#endif

