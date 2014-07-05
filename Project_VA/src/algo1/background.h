#ifndef AVGSEG_
#define AVGSEG_

#include "cv.h"
#include "highgui.h"
#include "cxcore.h"

#define NUM_CAMERAS   1
#define HIGH_SCALE_NUM 7.0
#define LOW_SCALE_NUM 6.0

void AllocateImages(IplImage *I);
void DeallocateImages();
void accumulateBackground(IplImage *I, int number=0);
void scaleHigh(float scale = HIGH_SCALE_NUM, int num = 0);
void scaleLow(float scale = LOW_SCALE_NUM, int num = 0);
void createModelsfromStats();
void backgroundDiff(IplImage *I,IplImage *Imask, int num = 0);

#endif

