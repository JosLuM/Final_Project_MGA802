Image Alignment and Comparison Tool
This tool provides functionality for aligning and comparing images using various techniques, such as rotation, feature matching, and metric calculation.

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
Import the necessary modules and classes from the tool into your Python code.

import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from image_tool import ImageCreator, ImageRotator, ImageAligner, ImageComparer
Use the provided classes and methods to perform image alignment and comparison tasks.

ImageCreator: Create and manipulate images for testing purposes (uses rosta_speckle function with parameters:dot_size=4, density=0.32, smoothness=2.0).
ImageRotator: Rotate images by a specified angle.
ImageAligner: Align two images using feature matching and affine transformation.
ImageComparer: Compare images using metrics such as Mean Square Error (MSE) and residuals.


Example Usage
Here is an example of how to use the tool in your own Python code:

python
Copy code
# Import the required modules and classes
import cv2
import imutils
from image_tool import ImageCreator, ImageRotator, ImageAligner, ImageComparer

# Create an instance of the ImageCreator class
image_creator = ImageCreator((2000, 2000))

# Create a speckle image
speckle_image = image_creator.create_speckle_image()

# Create an instance of the ImageRotator class
rotator = ImageRotator(speckle_image)

# Rotate the image by 30 degrees
rotated_image = rotator.rotate_image(30)

# Create an instance of the ImageAligner class
aligner = ImageAligner()

# Align the rotated image with the original speckle image
aligned_image = aligner.align_images(speckle_image, rotated_image)

# Create an instance of the ImageComparer class
comparer = ImageComparer()

# Calculate the Mean Square Error (MSE) between the aligned image and the original speckle image
mse = comparer.calculate_mse(speckle_image, aligned_image)

# Calculate the residuals between the aligned image and the original speckle image
residuals = comparer.calculate_residuals(speckle_image, aligned_image)

# Display the aligned image and the comparison results
comparer.display_comparison(speckle_image, aligned_image, mse, residuals)

Note: The code assumes that you have the necessary image files or have access to a camera to capture the images for the tests. 
The main function includes embedding rotated image to the canvas and saving it.
The code assumes that user has to follow the next logic:
First step:
run the code
choose option 1 : 
-speckle image created
-speckle image saved
-speckle image displayed
-speckle image rotated to 30 degrees
-rotated image saved for visualization
-rotated image aligned with the original speckle image
-comparison performed between aligned image and original speckle image
-marked image created for printing purpose and for further tests
-user analyze the result
Second step:
User prints marked image 
User takes pictures of marked image using markers as top-bottom reference, with different conditions, lighting, etc
User saves all captured images 
Third step:
User runs code for each captured image to be compared with the original digital marked image.
User analyzes the result for each comparison

