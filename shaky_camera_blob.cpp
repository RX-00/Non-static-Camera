/*
  NOTE: This program attempted to impliment some blob detection libraries
  that ultimately did not work out that well. So, I ended up writing my
  own in later iterations of this program seen in (in order of creation)
  -shaky_camera_edgeAndColor.cpp
  -shaky_camera_HSV_BGR.cpp
  -simple_blob_tracking.cpp
*/

// C/C++ stuff
#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fstream>
// OpenCV stuff
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video.hpp>
// OpenCVBlobsLib stuff
#include "BlobContour.h"
#include "blob.h"
#include "BlobOperators.h"
#include "BlobResult.h"

using namespace cv;
using namespace std;

void analyzeVideo();

int thresh = 100;
int max_thresh = 255;

//function to read the shaky video and object track
void analyzeVideo(){

  VideoCapture capVideo;

  //NOTE: the video must be in the same directory as the code is located in OR just link to it like /home/etc/...
  capVideo.open("09152008flight2tape2_1s_1.mp4"); //this is the video to be opened

  //checks if the video can be opened or not
  if(!capVideo.isOpened()){
    cout<<"error: cannot open video file to be read \n"<<endl;
    exit(EXIT_FAILURE);
  }

  //checks to make sure the video file has more than one frame to read
  if(capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 1){
    cout<<"error: the video file must have at least one frame in order to be analyzed"<<endl;
    exit(EXIT_FAILURE);
  }

  //read the captured video's frames by capturing a temporary image from the video
  Mat imgFrame; //for contour
  Mat original; //for everything
  Mat img_with_keypoints; //for keypoint tracking

  capVideo.read(imgFrame);
  capVideo.read(original);

  char chCheckForEscKey = 0;

  while(capVideo.isOpened() && chCheckForEscKey != 27){

    //if there is at least one frame then read it
    if((capVideo.get(CV_CAP_PROP_POS_FRAMES) +1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)){
      capVideo.read(imgFrame);
      capVideo.read(original);

      //Set up the blob detector with default parameters
      //SimpleBlobDetector detector;


      //Set up the blob detector parameters
      SimpleBlobDetector::Params params;

      //Change the thresholds
      params.minThreshold = 10;
      params.maxThreshold = 200;

      /*
      //Filter by Color
      params.filterByColor = true;
      params.blobColor = 0; //0 for darker blobs, 255 for lighter blobs
      */

      //Filter by Area
      params.filterByArea = true; //may need to be false since I want to detect background blobs or have a different SimpleBlobDetector to do that
      params.minArea = 1500; //filter out blobs that have less than 1500 pixels
      params.maxArea = 2000; //filter out blobs that have more than 2000 pixels

      //Filter by Circularity
      params.filterByCircularity = true;
      params.minCircularity = 0.1;

      //Filter by Convexity
      params.filterByConvexity = true; //probably needs not to be true for background blobs
      params.minConvexity = 0.1;

      //Filter by Inertia
      params.filterByInertia = true;
      params.minInertiaRatio = 0.01;

      SimpleBlobDetector detector(params);


      Mat edges;
      capVideo >> edges; //get a new frame from the video

      //Storage for the blobs
      vector <KeyPoint> keypoints;
      //Detect the blobs
      detector.detect(imgFrame, keypoints);

      //Draw detected blobs as blue circles
      drawKeypoints(imgFrame, keypoints, img_with_keypoints, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS); //DrawMatchesFlags make sure that the size of the circle corresponds to the size of the detected blob

      /*
      //convert the frame to gray
      cvtColor(edges, imgFrame, COLOR_BGR2GRAY);
      GaussianBlur(imgFrame, imgFrame, Size(7,7), 1.5, 1.5);

      Mat threshold_output;
      vector<vector<Point> > contours;
      vector<Vec4i> hierarchy;

      //detect edges using Threshold
      threshold(imgFrame, threshold_output, thresh, 255, THRESH_BINARY);
      //Find the contours
      findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));

      //Find the rectangles for each contour
      vector<RotatedRect> minRect(contours.size());

      for(int i = 0; i < contours.size(); i++){
        minRect[i] = minAreaRect(Mat(contours[i]));
      }

      //draw the rectangles
      Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
      for(int i = 0; i < contours.size(); i++){
        Scalar color = Scalar(0,0,255); //this makes the rectangles red
        //draw rectangles
        Point2f rect_points[4]; minRect[i].points(rect_points);
        for( int j = 0; j < 4; j++ ){
          line(drawing, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
        }
      }
      */

      //show modified videos
      /*
      imshow("Contours", drawing);
      imshow("Contours && Keypoint tracking", drawing+img_with_keypoints);
      */
      imshow("Keypoint tracking", img_with_keypoints);
      imshow("Original Video", original);
      waitKey(100);//opencv function which displays each frame for () milliseconds
    }
    else{
      cout<<"This is now the end of the video \n"<<endl;
      cout<<"Please press 'esc' in order to exit \n"<<endl;
      break;
    }

    //get the key press in case the user pressed esc
    chCheckForEscKey = waitKey(1);
  }
  //if the user did not press esc, aka the video ended
  if(chCheckForEscKey != 27){
    //hold the windows open so that the video end message has time to appear
    waitKey(0);
  }
  exit(EXIT_FAILURE);
}


int main(int argc, char** argv){
  analyzeVideo();
  return 0;
}
