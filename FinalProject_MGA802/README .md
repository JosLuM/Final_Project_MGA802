Image Alignment and Comparison Tool
This tool allows users to align and compare images for various purposes, such as testing digital images or real camera images. It provides functionality for image rotation, alignment, and comparison using different metrics.

-Prerequisites
Python 3.x
OpenCV
NumPy
imutils
matplotlib
muDIC

-Installation
Clone the repository or download the source code files.

Install the required dependencies using the following command:

pip install opencv-python numpy imutils matplotlib

-Usage
Place the images you want to work with in a same directory as your project containing the main.py file
Run the file

Follow the on-screen instructions and choose an option:

Option-1:Digital tests: Perform image creation,rotation, alignment and comparison on digital images, creates markers on image for printing purpose.
Option-2:Real camera tests: Perform alignment and comparison on images captured by a real camera(file_name=captured_image.jpg) to a base-line(digital image with markers).


View the output, which includes aligned images, comparison metrics, and visualizations.

Example Usage
Here is an example of how to use the tool for aligning and comparing digital test images:

Choose the "Digital tests" option.
Follow the on-screen instructions to view the creted scaled image,rotated image, aligned image and comparison results, such as Mean Square Error (MSE) and residuals.
Use printed_image.jpg file to print and capture this picture with different lighting conditions. Save all the files.

Run the code and Choose the "Real camera tests" option as many times as many captured files you have. Each time you run the code save the image you want to compare with the original image as captured_image.jpg
Follow the on-screen instructions to see the results 
