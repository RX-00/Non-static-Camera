//This program is from Haris, answers.opencv.org, just some stuff changed around and modified to my style (Roy Xing)
//The program is meant to choose an HSV value easily for things such as HSV color segmentation

#include <iostream>
#include <opencv2/opencv.hpp>
#include "stdio.h"

#define HUEMAX 179 //Hue
#define SATMAX 255 //Saturation
#define VALMAX 255 //Value

using namespace cv;
using namespace std;

/*-------------------------------------------------
  NOTE: The reason that there's some instances in
  which you call cv:: is because you, Roy Xing
  (oh ho what a smart guy), wanted to not use
  namespace to "differentiate." The program's not
  even more than 500 lines jackass

  August 9, 2016
  -Roy Xing
  --------------------------------------------------*/

Mat HSV;
int H = 170;
int S = 200;
int V = 200;

int R = 0;
int G = 0;
int B = 0;

int MAX_HUE = 179;
int MAX_SAT = 255;
int MAX_VAL = 255;

int mouse_x = 0;
int mouse_y = 0;

char window_name[20] = "HSV Color Plot";

//Global variables for the HSV color wheel plot
int max_hue_range = 179;
int max_step = 3; // number of pixels for each hue color
int wheel_width = max_hue_range*max_step;
int wheel_height = 50;
int wheel_x = 50; //x-position of the wheel
int wheel_y = 50; //y-position of the wheel

//Global variables for the plot of the saturation value plot
int S_V_Width = MAX_SAT;
int S_V_Height = MAX_SAT;
int S_V_x = 10;
int S_V_y = wheel_y + wheel_height + 20;

//Global variables for the HSV plot
int HSV_Width = 150;
int HSV_Height = 150;
int HSV_x = S_V_x + S_V_Width + 30;
int HSV_y = S_V_y + 50;

void onTrackbar_change(int, void*); //the void* is here for callback definitions, so the callback can receive user data of any type, including already defined object/structs (which should be casted to the appropiate type before using void*)
static void onMouse(int event, int x, int y, int, void*);
void drawPointers(void);


void onTrackbar_change(int, void*){
  //Plot the color wheel
  int hue_range = 0;
  int step = 1;
  for(int i = wheel_y; i < wheel_height + wheel_y; i++){
    hue_range = 0;
    for(int j = wheel_x; j < wheel_width + wheel_x; j++){
      if(hue_range >= max_hue_range){
        hue_range = 0;
      }
      if(step++ == max_step){
        hue_range++;
        step = 1;
      }
      Vec3b pix;
      pix.val[0] = hue_range;
      pix.val[1] = 255;
      pix.val[2] = 255;

      HSV.at<Vec3b>(i,j) = pix;
    }
  }

  //Plot for saturation and value
  int sat_range = 0;
  int value_range = 255;
  for(int i = S_V_y; i < S_V_Height + S_V_y; i++){
    value_range--;
    sat_range = 0;
    for(int j = S_V_x; j < S_V_Width + S_V_x; j++){
      cv::Vec3b pix;
      pix.val[0] = H;
      pix.val[1] = sat_range++;
      pix.val[2] = value_range;
      HSV.at<Vec3b>(i,j) = pix;
    }
  }

  //Plot for HSV
  cv::Mat ROI1(HSV, cv::Rect(HSV_x, HSV_y, HSV_Width, HSV_Height));
  ROI1 = cv::Scalar(H,S,V);
  drawPointers();

  cv::Mat RGB;
  cv::cvtColor(HSV, RGB, CV_HSV2BGR);

  cv::imshow(window_name, RGB);
  cv::imwrite("hsv.jpg", RGB);
}

static void onMouse(int event, int x, int y, int f, void*){
  if(f&CV_EVENT_FLAG_LBUTTON){
    mouse_x = x;
    mouse_y = y;
    if(((wheel_x <= x) && (x <= wheel_x + wheel_width)) && ((wheel_y <= y) && (y <= wheel_y + wheel_height))){
      H = (x - wheel_x) / max_step;
      cvSetTrackbarPos("Hue", window_name, H);
    }
    else if(((S_V_x <= x) && (x <= S_V_x + S_V_Width)) && ((S_V_y <= y) && (y <= S_V_y + S_V_Height))){
      S = x - S_V_x;
      y = y - S_V_y;
      V = 255 - y;

      cvSetTrackbarPos("Saturation", window_name, S);
      cvSetTrackbarPos("Value", window_name, V);
    }
  }
}

void drawPointers(){
  cv::Point p(S, 255 - V);

  int index = 10;
  cv::Point p1, p2;
  p1.x = p.x - index;
  p1.y = p.y;
  p2.x = p.x + index;
  p2.y = p.y;

  cv::Mat ROI1(HSV, cv::Rect(S_V_x, S_V_y, S_V_Width, S_V_Height));
  cv::line(ROI1, p1, p2, cv::Scalar(255,255,255), 1, CV_AA, 0);
  p1.x = p.x;
  p1.y = p.y - index;
  p2.x = p.x;
  p2.y = p.y + index;
  cv::line(ROI1, p1, p2, cv::Scalar(255,255,255), 1, CV_AA, 0);

  int x_index = wheel_x + H * max_step;
  if(x_index >= wheel_x + wheel_width){
    x_index = wheel_x + wheel_width - 2;
  }
  if(x_index <= wheel_x){
    x_index = wheel_x + 2;
  }

  p1.x = x_index;
  p1.y = wheel_y + 1;
  p2.x = x_index;
  p2.y = p.y + 20;
  cv::line(ROI1, p1, p2, cv::Scalar(255,255,255), 1, CV_AA, 0);

  cv::Mat RGB(1, 1, CV_8UC3);
  cv::Mat temp;
  RGB = cv::Scalar(H,S,V);
  cv::cvtColor(RGB, temp, CV_HSV2BGR);
  Vec3b rgb = temp.at<Vec3b>(0,0);
  B = rgb.val[0];
  G = rgb.val[1];
  R = rgb.val[2];

  cv::Mat ROI2(HSV, cv::Rect(450, 130, 175, 175));
  ROI2 = cv::Scalar(200,0,200);

  char name[30];
  sprintf(name, "R=%d", R);
  putText(HSV, name, cv::Point(460, 155), cv::FONT_HERSHEY_SIMPLEX, .7, cv::Scalar(5,255,255), 2, 8, false);

  sprintf(name, "G=%d", G);
  putText(HSV, name, cv::Point(460, 180), cv::FONT_HERSHEY_SIMPLEX, .7, cv::Scalar(5,255,255), 2, 8, false);

  sprintf(name, "B=%d", B);
  putText(HSV, name, cv::Point(460, 205), cv::FONT_HERSHEY_SIMPLEX, .7, cv::Scalar(5,255,255), 2, 8, false);

  sprintf(name, "H=%d", H);
  putText(HSV, name, cv::Point(545, 155), cv::FONT_HERSHEY_SIMPLEX, .7, cv::Scalar(5,255,255), 2, 8, false);

  sprintf(name, "S=%d", S);
  putText(HSV, name, cv::Point(545, 180), cv::FONT_HERSHEY_SIMPLEX, .7, cv::Scalar(5,255,255), 2, 8, false);

  sprintf(name, "V=%d", V);
  putText(HSV, name, cv::Point(545, 205), cv::FONT_HERSHEY_SIMPLEX, .7, cv::Scalar(5,255,255), 2, 8, false);
}


int main(){
  HSV.create(390, 640, CV_8UC3); //Mat to score the clock image
  HSV.setTo(cv::Scalar(200,0,200));

  cv::namedWindow(window_name);
  cv::createTrackbar("Hue", window_name, &H, HUEMAX, onTrackbar_change);
  cv::createTrackbar("Saturation", window_name, &S, SATMAX, onTrackbar_change);
  cv::createTrackbar("Value", window_name, &V, VALMAX, onTrackbar_change);
  onTrackbar_change(0,0); //initialize the window

  setMouseCallback(window_name, onMouse, 0);
  while(true){
    int c;
    c = waitKey(20);
    if( (char)c == 27 ){
      break;
    }
  }
  return 0;
}
