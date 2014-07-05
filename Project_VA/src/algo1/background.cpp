#include "background.h"

IplImage *IavgF[NUM_CAMERAS], *IdiffF[NUM_CAMERAS], *IprevF[NUM_CAMERAS],
		*IhiF[NUM_CAMERAS], *IlowF[NUM_CAMERAS];
IplImage *Iscratch, *Iscratch2, *Igray1, *Igray2, *Igray3, *Imaskt;
IplImage *Ilow1[NUM_CAMERAS], *Ilow2[NUM_CAMERAS], *Ilow3[NUM_CAMERAS],
		*Ihi1[NUM_CAMERAS], *Ihi2[NUM_CAMERAS], *Ihi3[NUM_CAMERAS];

float Icount[NUM_CAMERAS];

void AllocateImages(IplImage *I) {
	for (int i = 0; i < NUM_CAMERAS; i++) {
		IavgF[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3);
		IdiffF[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3);
		IprevF[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3);
		IhiF[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3);
		IlowF[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3);
		Ilow1[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
		Ilow2[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
		Ilow3[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
		Ihi1[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
		Ihi2[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
		Ihi3[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
		cvZero(IavgF[i]);
		cvZero(IdiffF[i]);
		cvZero(IprevF[i]);
		cvZero(IhiF[i]);
		cvZero(IlowF[i]);
		Icount[i] = 0.00001;
	}
	Iscratch = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3);
	Iscratch2 = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3);
	Igray1 = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
	Igray2 = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
	Igray3 = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
	Imaskt = cvCreateImage(cvGetSize(I), IPL_DEPTH_8U, 1);
	cvZero(Iscratch);
	cvZero(Iscratch2);
}

void DeallocateImages() {
	for (int i = 0; i < NUM_CAMERAS; i++) {
		cvReleaseImage(&IavgF[i]);
		cvReleaseImage(&IdiffF[i]);
		cvReleaseImage(&IprevF[i]);
		cvReleaseImage(&IhiF[i]);
		cvReleaseImage(&IlowF[i]);
		cvReleaseImage(&Ilow1[i]);
		cvReleaseImage(&Ilow2[i]);
		cvReleaseImage(&Ilow3[i]);
		cvReleaseImage(&Ihi1[i]);
		cvReleaseImage(&Ihi2[i]);
		cvReleaseImage(&Ihi3[i]);
	}
	cvReleaseImage(&Iscratch);
	cvReleaseImage(&Iscratch2);
	cvReleaseImage(&Igray1);
	cvReleaseImage(&Igray2);
	cvReleaseImage(&Igray3);
	cvReleaseImage(&Imaskt);
}

void accumulateBackground(IplImage *I, int number) {
	static int first = 1;
	cvCvtScale(I, Iscratch, 1, 0);
	if (!first) {
		cvAcc(Iscratch, IavgF[number]);
		cvAbsDiff(Iscratch, IprevF[number], Iscratch2);
		cvAcc(Iscratch2, IdiffF[number]);
		Icount[number] += 1.0;
	}
	first = 0;
	cvCopy(Iscratch, IprevF[number]);
}

void scaleHigh(float scale, int num) {
	cvConvertScale(IdiffF[num], Iscratch, scale);
	cvAdd(Iscratch, IavgF[num], IhiF[num]);
	cvCvtPixToPlane(IhiF[num], Ihi1[num], Ihi2[num], Ihi3[num], 0);
}

void scaleLow(float scale, int num) {
	cvConvertScale(IdiffF[num], Iscratch, scale);
	cvSub(IavgF[num], Iscratch, IlowF[num]);
	cvCvtPixToPlane(IlowF[num], Ilow1[num], Ilow2[num], Ilow3[num], 0);
}

void createModelsfromStats() {
	for (int i = 0; i < NUM_CAMERAS; i++) {
		cvConvertScale(IavgF[i], IavgF[i], (double) (1.0 / Icount[i]));
		cvConvertScale(IdiffF[i], IdiffF[i], (double) (1.0 / Icount[i]));
		cvAddS(IdiffF[i], cvScalar(1.0, 1.0, 1.0), IdiffF[i]);
		scaleHigh(HIGH_SCALE_NUM, i);
		scaleLow(LOW_SCALE_NUM, i);
	}
}

void backgroundDiff(IplImage *I, IplImage *Imask, int num) {
	cvCvtScale(I, Iscratch, 1, 0);
	cvCvtPixToPlane(Iscratch, Igray1, Igray2, Igray3, 0);
	cvInRange(Igray1, Ilow1[num], Ihi1[num], Imask);
	cvInRange(Igray2, Ilow2[num], Ihi2[num], Imaskt);
	cvOr(Imask, Imaskt, Imask);
	cvInRange(Igray3, Ilow3[num], Ihi3[num], Imaskt);
	cvOr(Imask, Imaskt, Imask);
	cvSubRS(Imask, cvScalar(255), Imask);
}
