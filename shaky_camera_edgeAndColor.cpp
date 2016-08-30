//TODO: Get and track background objects
//TODO: Fix your messy rectangle drawing BY FIXING THE FREAKING CONTOURS, may need to look at light stuff
//TODO: Do not track target objects that are HUGE (the rectangles don't say so, but the contours do. Try to watch each HSV seperately to make sure)
//TODO: Be careful of shadows, they are detected and are moving around when you don't want them to, thus becoming a target object that does not stay in place relative to background obejects -> it will be tracked

/*--------------------------------------------------------------
  NOTE: This program split up the video feed into the Hue,
  Saturation, and Value channels each for blob tracking. In
  simple_blob_tracking.cpp and shaky_camera_edgeAndColor.cpp I
  decided against this. Instead I simply just changed the video
  feed into HSV in order to cut down on processing and keep
  things more efficient and compact
 ---------------------------------------------------------------*/

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

using namespace cv;
using namespace std;

void analyzeVideo();

//-------------------------------------------------------------------------------------------------------------------------

//function to read the shaky video and object track
void analyzeVideo(){

  VideoCapture capVideo;

  //NOTE: the video must be in the same directory as the code is located in OR just link to it like /home/etc/...
  capVideo.open("09152008flight2tape2_1s_1.mp4"); //this is the video to be opened

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
  Mat imgFrame;
  Mat original;
  Mat img_with_keypoints;
  Mat HSV;

  capVideo.read(imgFrame);
  capVideo.read(original);

  char chCheckForEscKey = 0;

  while(capVideo.isOpened() && chCheckForEscKey != 27){

    //if there is at least one frame then read it
    if((capVideo.get(CV_CAP_PROP_POS_FRAMES) +1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)){
      capVideo.read(imgFrame);
      capVideo.read(original);

      Mat edges;
      capVideo >> edges; //get a new frame from the video

      //convert the frame to gray
      cvtColor(edges, imgFrame, COLOR_BGR2GRAY);
      //blur the image so that it will be easier to detect objects, or rather the edges
      GaussianBlur(imgFrame, imgFrame, Size(7,7), 1.5, 1.5);

      //convert the frame to HSV (Hue, Saturation, Value), HSV handles the "color" of an image directly instead of dividing it up into RGB components, this makes it easier to deal with different brightness
      cvtColor(edges, HSV, CV_BGR2HSV);

      vector<cv::Mat> channels;
      split(HSV, channels);

      Mat H = channels[0];
      Mat S = channels[1];
      Mat V = channels[2];

      //canny edge detector for edges in the Hue channel
      Mat cannyH; //really good for red car
      Canny(H, cannyH, 100, 50);
      //canny edge detector for edges in the Saturation channel
      Mat cannyS;
      Canny(S, cannyS, 100, 50);
      //canny edge detector for edges in the Value channel
      Mat cannyV;
      Canny(V, cannyV, 100, 50);

      //extract contours from the canny image
      vector<vector<Point> > contoursH; //vectors are nice here due to being like arrays that can change size dynamically
      vector<Vec4i> hierarchyH;
      vector<vector<Point> > contoursS;
      vector<Vec4i> hierarchyS;
      vector<vector<Point> > contoursV;
      vector<Vec4i> hierarchyV;
      findContours(cannyH, contoursH, hierarchyH, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
      findContours(cannyS, contoursS, hierarchyS, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
      findContours(cannyV, contoursV, hierarchyV, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

      //Combine %% Approximate Contours

      //Find the rectangles for each contour
      vector<RotatedRect> minRectH(contoursH.size());
      for(int i = 0; i < contoursH.size(); i++){
        minRectH[i] = minAreaRect(Mat(contoursH[i]));
      }
      vector<RotatedRect> minRectS(contoursS.size());
      for(int i = 0; i < contoursS.size(); i++){
        minRectS[i] = minAreaRect(Mat(contoursS[i]));
      }
      vector<RotatedRect> minRectV(contoursV.size());
      for(int i = 0; i < contoursV.size(); i++){
        minRectV[i] = minAreaRect(Mat(contoursV[i]));
      }

      //draw the rectangles for Hue
      Mat drawingRectanglesH = Mat::zeros(cannyH.size(), CV_8UC3);
      for(int i = 0; i < contoursH.size(); i++){
        Scalar color = Scalar(0,0,255); //this makes the rectangles red
        //draw rectangles
        Point2f rect_pointsH[4]; minRectH[i].points(rect_pointsH);
        for( int j = 0; j < 4; j++ ){
          line(drawingRectanglesH, rect_pointsH[j], rect_pointsH[(j+1)%4], color, 1, 8 );
        }
      }

      //draw the rectangles for Saturation
      Mat drawingRectanglesS = Mat::zeros(cannyS.size(), CV_8UC3);
      for(int i = 0; i < contoursS.size(); i++){
        Scalar color = Scalar(0,0,255); //this makes the rectangles red
        //draw rectangles
        Point2f rect_pointsS[4]; minRectS[i].points(rect_pointsS);
        for( int j = 0; j < 4; j++ ){
          line(drawingRectanglesS, rect_pointsS[j], rect_pointsS[(j+1)%4], color, 1, 8 );
        }
      }

      //draw the rectangles for Value
      Mat drawingRectanglesV = Mat::zeros(cannyV.size(), CV_8UC3);
      for(int i = 0; i < contoursV.size(); i++){
        Scalar color = Scalar(0,0,255); //this makes the rectangles red
        //draw rectangles
        Point2f rect_pointsV[4]; minRectV[i].points(rect_pointsV);
        for( int j = 0; j < 4; j++ ){
          line(drawingRectanglesV, rect_pointsV[j], rect_pointsV[(j+1)%4], color, 1, 8 );
        }
      }

      //draw the contours
      Mat HSVdrawing = edges.clone();
      for(int i = 0; i < contoursH.size(); i++){
        drawContours(HSVdrawing, contoursH, i, Scalar(0,0,255), 2, 8, hierarchyH, 0);
      }
      for(int i = 0; i < contoursS.size(); i++){
        drawContours(HSVdrawing, contoursS, i, Scalar(0,0,255), 2, 8, hierarchyS, 0);
      }
      for(int i = 0; i < contoursV.size(); i++){
        drawContours(HSVdrawing, contoursV, i, Scalar(0,0,255), 2, 8, hierarchyV, 0);
      }


      //show  videos
      imshow("Just tracking with rectangle drawing", drawingRectanglesH+drawingRectanglesV+drawingRectanglesS);
      imshow("Track + Drawing rectangles HSV", original+drawingRectanglesH+drawingRectanglesV+drawingRectanglesS);
      imshow("Contours", HSVdrawing-original);

      /* shorten the code by keeping HSV together, probably better too, but first deal with the contour problem, then you can decide if you want to split everything up or keep it all together

         -Roy Xing June 30, 2016
        ----------------------------------------------------------------------------------------------------------------------- 
      //try to combine HSV
      Mat cannyHSV;
      Canny(HSV, cannyHSV, 100, 50);

      vector<vector<Point> > contoursHSV;
      vector<Vec4i> hierarchyHSV;
      findContours(cannyHSV, contoursHSV, hierarchyHSV, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
      vector<RotatedRect> minRectHSV(contoursHSV.size());
      for(int i = 0; i < contoursHSV.size(); i++){
        minRectHSV[i] = minAreaRect(Mat(contoursHSV[i]));
      }

      Mat drawingRectanglesHSV = Mat::zeros(cannyHSV.size(), CV_8UC3);
      for(int i = 0; i < contoursHSV.size(); i++){
        Scalar color = Scalar(0,0,255); //this makes the rectangles red
        }
        //draw rectangles
        Point2f rect_pointsHSV[4]; minRectHSV[i].points(rect_pointsHSV);
        for( int j = 0; j < 4; j++ ){
          line(drawingRectanglesHSV, rect_pointsHSV[j], rect_pointsHSV[(j+1)%4], color, 1, 8 );
        }
      }

      Mat HSVdrawing = edges.clone();
      for(int i = 0; i < contoursHSV.size(); i++){
        drawContours(HSVdrawing, contoursHSV, i, Scalar(0,0,255), 2, 8, hierarchyHSV, 0);
      }
      imshow("Track with rectangles", drawingRectanglesHSV+original);
      imshow("Just rectangles", drawingRectanglesHSV);
      imshow("Contours from HSV", HSVdrawing-original);
      --------------------------------------------------------------------------------------------------------------------------
        */
      imshow("Original Video", original);
      waitKey(500);//opencv function which displays each frame for () milliseconds
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
