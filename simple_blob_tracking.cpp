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
#include <vector>
#include <algorithm>
#include <iterator>
// OpenCV stuff
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;

//parameters for the colors being tracking
//NOTE: These were not implimented yet as the program is incomplete -Roy Xing August 11, 2016
#define MIN_H_RED 0
#define MAX_H_RED 70
#define MIN_H_GREEN 200
#define MAX_H_GREEN 300
#define MIN_H_WHITE 200
#define MAX_H_WHITE 300
#define MIN_H_SILVER 200
#define MAX_H_SILVER 300

//Variables for Red Blob Detection
//NOTE: these values are not finalized, feel free to change them
int LowH = 0;
int HighH = 179;

int LowS = /*45*/ 135;
int HighS = 255;

int LowV = /*110*/ 165;
int HighV = 255;

//Previous Points for the Color Masking object
int lastX = -1;
int lastY = -1;

void analyzeVideo();
float distanceFormula();
void kalmanFilter();
void createTrackbars();

//-------------------------------------------------------------------------------------------------------------------------

//Euclidian Distance Formula
float distanceFormula(float x1, float x2, float y1, float y2){
  float distance;
  distance = sqrt(pow((x1-x2), 2) + pow((y1-y2), 2));
  return distance;
}

//trackbars for filtering red (meant for the Kalman Filter)
void createTrackbars(){
  namedWindow("Control Red Side", CV_WINDOW_AUTOSIZE);

  cvCreateTrackbar("LowH Side", "Control Red Side", &LowH, 179);
  cvCreateTrackbar("HighH Side", "Control Red Side", &HighH, 179);

  cvCreateTrackbar("LowS Side", "Control Red Side", &LowS, 255);
  cvCreateTrackbar("HighS Side", "Control Red", &HighS, 255);

  cvCreateTrackbar("LowV Side", "Control Red Side", &LowV, 255);
  cvCreateTrackbar("HighV Side", "Control Red Side", &HighV, 255);
}

//Kalman Filter Initialization
//NOTE: I later just initialized in the analyzeVideo function instead of calling this function, had a strange bug that I didn't have time to fix -Roy Xing August 11, 2016
void kalmanFilter(){
  int stateSize = 6;
  int measSize = 4;
  int contrSize = 0;

  cv::KalmanFilter kf(stateSize, measSize, contrSize, CV_32F);

  cv::Mat state(stateSize, 1, CV_32F); // [x,y,v_x,v_y,w,h]
  cv::Mat meas(measSize, 1, CV_32F); // [z_x,z_y,z_w,z_h]

  /*
    Transition State Matrix A
    NOTE: set dT at each processing step

    [1 , 0 , dT , 0  , 0 , 0 ]
    [0 , 1 , 0  , dT , 0 , 0 ]
    [0 , 0 , 1  , 0  , 0 , 0 ]
    [0 , 0 , 0  , 1  , 0 , 0 ]
    [0 , 0 , 0  , 0  , 1 , 0 ]
    [0 , 0 , 0  , 0  , 0 , 1 ]
  */
  cv::setIdentity(kf.transitionMatrix);

  /*
    Measure Matrix H
    [1 , 0 , 0 , 0  , 0 , 0 ]
    [0 , 1 , 0 , 0  , 0 , 0 ]
    [0 , 0 , 0 , 0  , 1 , 0 ]
    [0 , 0 , 0 , 0  , 0 , 1 ]
  */
  kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, CV_32F);
  kf.measurementMatrix.at<float>(0) = 1.0f;
  kf.measurementMatrix.at<float>(7) = 1.0f;
  kf.measurementMatrix.at<float>(16) = 1.0f;
  kf.measurementMatrix.at<float>(23) = 1.0f;

  /*
    Process Noise Covariance Matrix Q
    [Ex , 0 ,  0    ,  0  ,   0  , 0 ]
    [0  , Ey,  0    ,  0  ,   0  , 0 ]
    [0  , 0 , Ev_x  ,  0  ,   0  , 0 ]
    [0  , 0 ,  0    ,  1  , Ev_y , 0 ]
    [0  , 0 ,  0    ,  0  ,   1  , Ew ]
    [0  , 0 ,  0    ,  0  ,   0  , Eh ]
  */
  kf.processNoiseCov.at<float>(0) = 1e-2;
  kf.processNoiseCov.at<float>(7) = 1e-2;
  kf.processNoiseCov.at<float>(14) = 2.0f;
  kf.processNoiseCov.at<float>(21) = 1.0f;
  kf.processNoiseCov.at<float>(28) = 1e-2;
  kf.processNoiseCov.at<float>(35) = 1e-2;

  //Measure Noise Covariance Matrix R
  cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
}

//function to read the shaky video and object track
void analyzeVideo(){
  VideoCapture capVideo;

  int whichVideo;
  string filename = "";

  cout<<"\n"<<"\n"<<"Please choose the number of the video you wish to analyze \n"<<
        "\n"<<
        "1: small people walking past each other on a big path \n"<<
        "2: white and red truck \n"<<
        "3: driver switching cars \n"<<
        "4: car driving backwards \n"<<
        "5: panning shot of building and people \n"<<
        "6: lots of moving people \n"<<
    endl;

  cin>>whichVideo;

  cout<<"You have choosen video \n"<<whichVideo<<endl;

  switch(whichVideo){
  case 1:
    //NOTE: the video must be in the same directory as the code is located in OR link to it
    capVideo.open("09152008flight2tape1_6s.mp4"); //this is the video to be opened
    break;
  case 2:
    capVideo.open("09152008flight2tape2_1s_1.mp4");
    break;
  case 3:
    capVideo.open("09152008flight2tape2_1s2_1.mp4");
    break;
  case 4:
    capVideo.open("09152008flight2tape3_1s.mp4");
    break;
  case 5:
    capVideo.open("09152008flight2tape3_2s.mp4");
    break;
  case 6:
    capVideo.open("09172008flight1tape1_5s.mp4");
    break;
  }

  /*
------------------------------------------------------------
Lists of videos:
This one has small people walking past each other who are standing in front of a well defined path, might have to subtract that color of path, or anything that is detected of that size
-09152008flight2tape1_6s.mp4

This is the video of the white and red truck, note the tiny people walking around, really hard to differentiate and track those people
-09152008flight2tape2_1s_1.mp4

This video is of a person switching cars, however the backgrounds are detected really well so it makes the contours really messy, perhaps use color seperation to distinguish between the target objects and the to be background objects
-09152008flight2tape2_1s2_1.mp4

Video of a car driving backwards, going to have to somehow define background objects, perhaps with color and size
-09152008flight2tape3_1s.mp4

Video of a panning shot with lots of little people running around, the size of the people, background object detection, and shadows are a problem here (though the most distinctive shadow follows a person)
-09152008flight2tape3_2s.mp4

Video of lots of people walking around and moving, though they are bigger and thus easier to detect here, the background object detection problems (or rather lack of background object detection) is a problem here
-09172008flight1tape1_5s.mp4
------------------------------------------------------------
  */

  //checks if video can be opened
  if(!capVideo.isOpened()){
    cout<<"error: cannot open video file to be read \n"<<endl;
    exit(EXIT_FAILURE);
  }

  //create trackbars for the Kalman Filter
  createTrackbars();

  //makes sure the video has more than 1 frame
  if(capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 1){
    cout<<"error: the video file must have at least 1 frame in order to be analyzed"<<endl;
    exit(EXIT_FAILURE);
  }

  //>>> Simple color thresholding---------------------------------------------
  createTrackbars();

  //tmp image from the video
  cv::Mat imgTmp;
  capVideo.read(imgTmp);

  //Create a black image with the same size of the video feed
  cv::Mat imgBlack = Mat::zeros(imgTmp.size(), CV_8UC3);;

  //<<< Simple color thresholding---------------------------------------------


  //create some matrices
  Mat imgFrame;
  Mat original;
  Mat tmpImg;
  Mat HSV;

  //Read the incoming frames from the video
  capVideo.read(imgFrame);
  capVideo.read(original);

  //The character value to detect when the user has pressed the "esc" key
  char chCheckForEscKey = 0;

  //this is used in order to record the prior distance to the current distance
  //These arrays hold the class data
  const size_t N = 2;
  int frameData[N];
  int prevframeData[N];

  //These vectors hold the arrays
  vector<int> recordData;
  vector<int> recordPrevData;
  //These vectors are used to find specific info from the vectors that hold the arrays
  vector<int>::iterator currentDistance = recordData.begin();
  vector<int>::iterator previousDistance = recordPrevData.begin();
  vector<int>::iterator currentPointOne = recordData.begin();
  vector<int>::iterator currentPointTwo = recordData.begin();
  vector<int>::iterator prevPointOne = recordPrevData.begin();
  vector<int>::iterator prevPointTwo = recordPrevData.begin();

  //These are meant for comparing the distances to determine if a distance between a couple of objects changed dramatically across different frames, which means the object whose distances changed a lot is between the objects around it is an actual moving object
  const int distanceThreshold = 100;
  int distanceDifference;

  //This class holds the currently detected first point, second point, and their distance between each other
  class frameinfo{
  public:
    int pointOne;
    int pointTwo;
    int distance;
  }frameInfo;

  //This class holds the previously detected first point, second point, and their distance between each other
  class prevframeinfo{
  public:
    int pointOne;
    int pointTwo;
    int distance;
  }prevFrameInfo;

  //call the Kalman filter here
  //kalmanFilter();
  int stateSize = 6;
  int measSize = 4;
  int contrSize = 0;

  cv::KalmanFilter kf(stateSize, measSize, contrSize, CV_32F);

  cv::Mat state(stateSize, 1, CV_32F); // [x,y,v_x,v_y,w,h]
  cv::Mat meas(measSize, 1, CV_32F); // [z_x,z_y,z_w,z_h]

  /*
    Transition State Matrix A
    NOTE: set dT at each processing step

    [1 , 0 , dT , 0  , 0 , 0 ]
    [0 , 1 , 0  , dT , 0 , 0 ]
    [0 , 0 , 1  , 0  , 0 , 0 ]
    [0 , 0 , 0  , 1  , 0 , 0 ]
    [0 , 0 , 0  , 0  , 1 , 0 ]
    [0 , 0 , 0  , 0  , 0 , 1 ]
  */
  cv::setIdentity(kf.transitionMatrix);

  /*
    Measure Matrix H
    [1 , 0 , 0 , 0  , 0 , 0 ]
    [0 , 1 , 0 , 0  , 0 , 0 ]
    [0 , 0 , 0 , 0  , 1 , 0 ]
    [0 , 0 , 0 , 0  , 0 , 1 ]
  */
  kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, CV_32F);
  kf.measurementMatrix.at<float>(0) = 1.0f;
  kf.measurementMatrix.at<float>(7) = 1.0f;
  kf.measurementMatrix.at<float>(16) = 1.0f;
  kf.measurementMatrix.at<float>(23) = 1.0f;

  /*
    Process Noise Covariance Matrix Q
    [Ex , 0 ,  0    ,  0  ,   0  , 0 ]
    [0  , Ey,  0    ,  0  ,   0  , 0 ]
    [0  , 0 , Ev_x  ,  0  ,   0  , 0 ]
    [0  , 0 ,  0    ,  1  , Ev_y , 0 ]
    [0  , 0 ,  0    ,  0  ,   1  , Ew ]
    [0  , 0 ,  0    ,  0  ,   0  , Eh ]
  */
  kf.processNoiseCov.at<float>(0) = 1e-2;
  kf.processNoiseCov.at<float>(7) = 1e-2;
  kf.processNoiseCov.at<float>(14) = 2.0f;
  kf.processNoiseCov.at<float>(21) = 1.0f;
  kf.processNoiseCov.at<float>(28) = 1e-2;
  kf.processNoiseCov.at<float>(35) = 1e-2;

  //Measure Noise Covariance Matrix R
  cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));

  //some variables for the kalman filter
  char ch = 0;
  double ticks = 0;
  bool found = false;
  int notFoundCount = 0;

  //main loop for analyzing the video feed
  while(capVideo.isOpened() && chCheckForEscKey != 27){
    //if there is at least one frame then read it
    if((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)){
      capVideo.read(imgFrame);
      capVideo.read(original);
      capVideo.read(tmpImg);

      //these are for creating a black image with the size the same of the video
      Mat imgLines = cv::Mat::zeros(tmpImg.size(), CV_8UC3);;

      //some more matrices to hold the video's frames' info
      cv::Mat edges;
      cv::Mat frame;

      //get the seconds for dT in the Kalman Filter
      double precTick = ticks;
      ticks = (double) cv::getTickCount();
      double dT = (ticks - precTick)/cv::getTickFrequency(); //seconds

      capVideo >> edges; //get a new frame from the Video
      capVideo >> frame;

      //>>> Kalman Filtering---------------------------------------------
      cv::Mat res;
      frame.copyTo(res);

      if(found){
        //send info to Matrix A
        kf.transitionMatrix.at<float>(2) = dT;
        kf.transitionMatrix.at<float>(9) = dT;

        //see what value dT is at
        cout<<"dT: "<<dT<<endl;

        //check out the state value
        state = kf.predict();
        cout<<"State post: "<<state<<endl;

        //predicted rectangle
        //NOTE: To be quite honest, I'm not 100% sure what this stuff does/is meant to do here, Kalman Filters are complicated... -Roy Xing August 11, 2016
        /*----------------------------------------------------------------------------------------
          refer to these websites for more info:
          http://www.robot-home.it/blog/en/software/ball-tracker-con-filtro-di-kalman/
          http://opencvexamples.blogspot.com/2014/01/kalman-filter-implementation-tracking.html
          http://www.morethantechnical.com/2011/06/17/simple-kalman-filter-for-tracking-using-opencv-2-2-w-code/
         ----------------------------------------------------------------------------------------*/
        cv::Rect predRect;
        predRect.width = state.at<float>(4);
        predRect.height = state.at<float>(5);
        predRect.x = state.at<float>(0) - predRect.width/2;
        predRect.y = state.at<float>(1) - predRect.height/2;
        cv::Point centerPredRect;
        centerPredRect.x = state.at<float>(0);
        centerPredRect.y = state.at<float>(1);
        cv::circle(edges, centerPredRect, 2, CV_RGB(255,0,0), -1);
        cv::rectangle(res, predRect, CV_RGB(255,0,0), 2);
      }

      //smooth out the noise
      cv::Mat blur;
      cv::GaussianBlur(frame, blur, cv::Size(5,5), 3.0, 3.0);
      //HSV conversion
      cv::Mat frameHSV;
      cv::cvtColor(blur, frameHSV, CV_BGR2HSV);
      //Color Thresholding *NOTE: this is for the red truck atm (August 9, 2016)
      cv::Mat rangeRes = cv::Mat::zeros(frame.size(), CV_8UC1);
      //cv::inRange(frameHSV, cv::Scalar(MIN_H_RED / 2, 100, 80), cv::Scalar(MAX_H_RED / 2, 255, 255), rangeRes);
      cv::inRange(frameHSV, cv::Scalar(LowH, LowS, LowV), cv::Scalar(HighH, HighS, HighV), rangeRes);
      //Improving/Cleaning up the result
      erode(rangeRes, rangeRes, cv::Mat(), cv::Point(-1, -1), 2);
      dilate(rangeRes, rangeRes, cv::Mat(), cv::Point(-1, -1), 2);

      //contour detection
      vector<vector<Point> > kalman_contours;
      cv::findContours(rangeRes, kalman_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

      //filtering
      vector<vector<Point> > blobs;
      vector<Rect> blobsBox;
      for(size_t i = 0; i < kalman_contours.size(); i++){
        cv::Rect bBox;
        bBox = cv::boundingRect(kalman_contours[i]);
        float ratio = (float) bBox.width/(float) bBox.height;
        if(ratio > 1.0f){
          ratio = 1.0f/ratio;
        }
        if(ratio > 0.75 && bBox.area() >= 400){ //search for a square bBox
          blobs.push_back(kalman_contours[i]);
          blobsBox.push_back(bBox);
        }
      }

      cout<<"Blobs Found: "<<blobsBox.size()<<endl;
      //Detection result
      for(size_t i = 0; i < blobs.size(); i++){
        cv::drawContours(res, blobs, i, CV_RGB(20, 150, 20), 1); //draws the contours on the video
      cv:rectangle(res, blobsBox[i], CV_RGB(0, 255, 0), 2); //draws the rectangles on the video
        cv::Point center; //gain the center of the boxes
        center.x = blobsBox[i].x + blobsBox[i].width / 2;
        center.y = blobsBox[i].y + blobsBox[i].height / 2;
        cv::circle(res, center, 2, CV_RGB(20, 150, 20), -1); //draw a circle in the center of each bounding box on the video

        stringstream sstr;
        sstr<<"("<<center.x<<", "<<center.y<<")";
        cv::putText(res, sstr.str(), cv::Point(center.x + 3, center.y - 3), cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(20, 150, 20), 2); //display the coordinates of each bounding box's center on the video
      }

      //Kalman Update
      if(blobs.size() == 0){
        notFoundCount++;
        cout<<"Not Found Count: "<<notFoundCount<<endl;
        if(notFoundCount >= 10){
          found = false;
        }
        else{
          kf.statePost = state;
        }
      }
      else{
        notFoundCount = 0;

        meas.at<float>(0) = blobsBox[0].x + blobsBox[0].width / 2;
        meas.at<float>(1) = blobsBox[0].y + blobsBox[0].height / 2;
        meas.at<float>(2) = (float)blobsBox[0].width;
        meas.at<float>(3) = (float)blobsBox[0].height;

        if(!found){ //first detection
          //initialization
          kf.errorCovPre.at<float>(0) = 1; // px
          kf.errorCovPre.at<float>(7) = 1; // px
          kf.errorCovPre.at<float>(14) = 1;
          kf.errorCovPre.at<float>(21) = 1;
          kf.errorCovPre.at<float>(28) = 1; // px
          kf.errorCovPre.at<float>(35) = 1; // px

          state.at<float>(0) = meas.at<float>(0);
          state.at<float>(1) = meas.at<float>(1);
          state.at<float>(2) = 0;
          state.at<float>(3) = 0;
          state.at<float>(4) = meas.at<float>(2);
          state.at<float>(5) = meas.at<float>(3);

          found = true;
        }
        else{
          kf.correct(meas); //Kalman correction
        }
        cout<<"Measure matrix: "<<meas<<endl;
      }
      //<<< Kalman Filtering---------------------------------------------

      //>>> Simple color thresholding------------------------------------
      //convert image from BGR to HSV
      cv::Mat imgHSV;
      cvtColor(imgTmp, imgHSV, COLOR_BGR2HSV);

      //Threshold for the red truck
      cv::Mat imgThresholded;
      cv::inRange(imgHSV, cv::Scalar(LowH, LowS, LowV), cv::Scalar(HighH, HighS, HighV), imgThresholded);

      //Morphological opening, remove small objects from foreground
      cv::erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
      cv::dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
      cv::erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
      cv::dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

      //Morphological closing, remove small holes from the foreground
      cv::dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
      cv::erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
      cv::dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
      cv::erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

      //Calculate the moments of the thresholded image in order to get the coordinates and size
      Moments oMoments = moments(imgThresholded);
      double dM01 = oMoments.m01;
      double dM10 = oMoments.m10;
      double dArea = oMoments.m00;

      //if the area of the detected blob is < 10000, then there is no object in the image due to noise
      if(dArea > 10000){
        //Calculate the position of the object
        int posX = dM10 / dArea;
        int posY = dM01 / dArea;

        if((lastX >= 0 && lastY >= 0 && posX >= 0 && posY >= 0)){
          //draw a red line from the previous point to the current point
          cv::line(imgBlack, Point(posX, posY), Point(lastX, lastY), cv::Scalar(255,0,0), 2);
        }

        //update the previous points
        lastX = posX;
        lastY = posY;
      }

      //<<< Simple color thresholding------------------------------------

      //NOTE: The red is the kalman filter prediction of where the object is, the green is where the object actually is

      //cv::imshow("name of the window", cv::Mat);
      //Essentially imshow displays a window of the matrices' video info
      imshow("Thresholded Image Kalman Filter", rangeRes);
      imshow("Kalman Filtering", res);

      imshow("Thresholded Image Color Masking", imgThresholded);
      imshow("Color Masking Result", imgBlack + original);

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
  //if the user did not press esc, and the video instead ended by itself
  if(chCheckForEscKey != 27){
    //hold the windows open so that the end message has time to appear
    waitKey(0);
  }
  exit(EXIT_FAILURE);
}


int main(int argc, char** argv){
  while(1){
    analyzeVideo();
  }
  return 0;
}
