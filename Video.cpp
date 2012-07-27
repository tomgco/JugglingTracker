#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <math.h>
#include <cvblob.h>
#include <vector>

using namespace cvb;
using namespace cv;
using namespace std;

/**
 *  Get the longest living track, this will most likley be the helicopter.
 */
bool longestLiving(const pair<CvID, CvTrack*>  &p1, const pair<CvID, CvTrack*> &p2) {
  return p1.second->lifetime < p2.second->lifetime;
}

int main() {
  CvTracks tracks;

  cvNamedWindow("Track", CV_WINDOW_AUTOSIZE);

  CvCapture *capture = cvCaptureFromCAM(0);
  cvGrabFrame(capture);
  IplImage *img = cvRetrieveFrame(capture);

  CvSize imgSize = cvGetSize(img);

  IplImage *frame = cvCreateImage(imgSize, img->depth, img->nChannels);

  IplConvKernel* morphKernel = cvCreateStructuringElementEx(5, 5, 1, 1, CV_SHAPE_RECT, NULL);

  vector <CvPoint> ballLines;

  while (cvGrabFrame(capture)) {
    IplImage *img = cvRetrieveFrame(capture);

    cvConvertScale(img, frame, 1, 0);

    IplImage *contrast = cvCreateImage(imgSize, 8, 1);

    // Detecting red pixels:
    // (This is very slow, use direct access better...)
    for (unsigned int j=0; j<imgSize.height; j++) {
      for (unsigned int i=0; i<imgSize.width; i++) {
        CvScalar c = cvGet2D(frame, j, i);
        double b = c.val[0];
        double g = c.val[1];
        double r = c.val[2];
        CvScalar found = CV_RGB(0, 0, 0);
        if ((r > 60 + b) && (g > 60 + b)) {
          found = CV_RGB(255, 255, 255);
        }
        cvSet2D(contrast, j, i, found);
      }
    }

    cvMorphologyEx(contrast, contrast, NULL, morphKernel, CV_MOP_OPEN, 1);

    IplImage *labelImg = cvCreateImage(cvGetSize(frame), IPL_DEPTH_LABEL, 1);

    CvBlobs blobs;
    unsigned int result = cvLabel(contrast, labelImg, blobs);
    cvFilterByArea(blobs, 500, 100000);
    cvRenderBlobs(labelImg, blobs, frame, frame, CV_BLOB_RENDER_BOUNDING_BOX);
    cvUpdateTracks(blobs, tracks, 200., 5);
    cvRenderTracks(tracks, frame, frame, CV_TRACK_RENDER_ID|CV_TRACK_RENDER_BOUNDING_BOX);


    vector <pair <CvLabel, CvTrack*> > trackList;
    copy(tracks.begin(), tracks.end(), back_inserter(trackList));

    sort(trackList.begin(), trackList.end(), longestLiving);

    if (trackList.size() > 0) {
      // This will print: [CvID] -> [x, y]
      ballLines.push_back(cvPoint(trackList[0].second->centroid.x, trackList[0].second->centroid.y));
      std::cout << trackList[0].first << " -> ["
        << (labelImg->width / 2 - (trackList[0].second->centroid.x))
        << ", " << (labelImg->height / 2 - (trackList[0].second->centroid.y))
        << "]" << endl;
    }

    if (ballLines.size() > 1) {
      for(int i=0;i < ballLines.size(); i++){
        if (i+1 < ballLines.size())
        cvLine(frame, ballLines.at(i), ballLines.at(i+1), CV_RGB(255, 0, 0), 3);
      }
      if (ballLines.at(0).y - 40 < ballLines.at(ballLines.size()-1).y) {
        ballLines.erase(ballLines.begin(), ballLines.end());
      }
    }


    cvShowImage("Track", frame);

    cvReleaseImage(&labelImg);
    cvReleaseImage(&contrast);

    cvReleaseBlobs(blobs);
  }

  cvReleaseStructuringElement(&morphKernel);
  cvReleaseImage(&frame);

  cvDestroyWindow("Track");

  return 0;
}
