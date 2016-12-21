# Non-static-Camera
object tracking with a non-static camera

C++ Compiler, g++ g++ pkg-config opencv --cflags shaky_camera_HSV_BGR.cpp -o shaky_camera_HSV_BGR.out pkg-config opencv --libs -I/usr/local/include/opencvblobslib -L/usr/local/lib -lopencvblobslib -std=c++11


These source code programs are the result of my (Roy Xing) summer internship at Aptima during the summer of 2016 (as a rising high school junior).

These programs use OpenCV 2.4.13 and the C++ standard library

The main goal of this project was to create a program that could track an object from a video feed of a non static
camera (simply just a very shaky camera). The videos are from the VIRAT public dataset.

The problem that arose from shaky camera feed is that normal methods for object tracking such as background
subtraction prove useless and or incredibly complicated to impliment. This is due to everything in the camera
feed appearing to be "moving" despite the "movement" of most objects is a result of the shaky camera.

As of writing this (August 11, 2016) near the end of my internship this problem is a very challenging issue
that is still daunting the computer vision field. (Even these sets of programs are incomplete in solving
this issue, though I will continue development).

The method that I present to solve this issue is to track all the objects in the video feed
(with methods such as Kalman Filters, Regions of Interest, color masking, or a combination of these)
and then compare the distances between all the objects in the video feed. All the distances are then
compared from the previous frame to the current frame in order to see if any distance between a set
of blobs/objects changed dramatically. If the distance is relatively the same then one can determine
that those objects are not moving. If the distance varies a lot then one can determine that object
is moving relative to the all the other objects and is thus a real moving object.

The programs in this folder have been created in this order and purpose:
1. shaky_camera_blob.cpp
This program contains a failed implimentation/usage of a blob detecting library that did not produce desirable results, so I made my own as seen in the later programs.

2. shaky_camera_edgeAndcolor.cpp
This program contains multiple channel HSV and contour blob detection. It was later replaced by the shaky_camera_HSV_BGR.cpp program as that program was more efficient.

3. shaky_camera_HSV_BGR.cpp
This program contains the Kalman Filter method, and HSV and contour blob detection. The distance calculations have been completed here, but is flawed.

4. simply_blob_tracking.cpp
This program contains the Kalman Filter method and color masking method for blob tracking, distance calculations have not been finished here.

5. HSV_Color_Plot.cpp
This program is simply meant to be a helpful tool to determine HSV range values.

Thank you for reading this README! I hope you found it useful, if you need any clarifications I (Roy Xing) can be contacted through
these emails:
00royxing@gmail.com
xing_roy@yahoo.com

Good Luck in your computer vision ventures!
