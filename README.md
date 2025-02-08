**CV & DL Project Report**

Problem Statement - Integrated System for Traffic Sign Recognition Using
YOLOv8 and CNN Models and lane detection using computer vision.
Domain – Intelligent Transport System

**Technologies Used:**
1. Traffic Sign Detection and Classification:
- YOLOv8: Used for object detection to extract traffic signs from the input
image.
- Convolutional Neural Network (CNN): Used for classifying the detected
traffic signs into specific categories.

2. Lane Detection: 
- Custom Code (as provided): Implemented for detecting lane markings in
the image using the following Computer Vision (CV) techniques:
- Grayscale Conversion: Simplifies image processing by reducing it to a
single intensity channel.
- Gaussian Blurring: Removes noise and smoothens the image for better
edge detection.
- Canny Edge Detection: Identifies edges in the image by detecting areas
with significant intensity gradients.
- Region of Interest Masking (ROI): Focuses on the area where lanes are
expected (e.g., the road).
- Hough Line Transformation: Detects lines in the masked edge image using
a probabilistic approach.
- Slope and Intercept Averaging: Groups lines into left and right lanes and
averages them to produce a smooth lane line representation.

This combination of deep learning and computer vision ensures accurate traffic
sign recognition and lane detection for enhanced road safety and automated
assistance. The code is particularly suited for detecting lanes that are
primarily straight, as the Hough Transform and averaging techniques are
optimized for such patterns.

**Input type** – RGB Image with dimension (32,32,3)

**Output –**
1. For traffic sign using cnn –
- Prediction for input image
- Prediction for extracted sign from input image
- Prediction for enhanced sign by CV techniques.
2. Lane detection using cv –
- Canny edge detection
- Region of Intrest
- Final lane
