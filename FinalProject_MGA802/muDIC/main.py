import muDIC.vlab as vlab
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils
import os


class ImageMarkerPrinter:
    def __init__(self, marker_color=(0, 0, 255), marker_thickness=5, marker_thickness_bottom=10):
        self.marker_color = marker_color
        self.marker_thickness = marker_thickness
        self.marker_thickness_bottom = marker_thickness_bottom

    def create_marker_image(self, img):
        # Get the dimensions of the image
        height, width, _ = img.shape

        # Create a blank canvas for the markers
        marker_img = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw markers on each corner
        marker_size = int(min(height, width) * 0.1)
        marker_thickness = self.marker_thickness
        marker_thickness_bottom = self.marker_thickness_bottom
        # Top-left corner marker
        marker_img = cv2.rectangle(marker_img, (0, 0), (marker_thickness, marker_size), self.marker_color, -1)
        marker_img = cv2.rectangle(marker_img, (0, 0), (marker_size, marker_thickness), self.marker_color, -1)

        # Top-right corner marker
        marker_img = cv2.rectangle(marker_img, (width - marker_thickness, 0), (width, marker_size), self.marker_color,
                                   -1)
        marker_img = cv2.rectangle(marker_img, (width - marker_size, 0), (width, marker_thickness), self.marker_color,
                                   -1)

        # Bottom-left corner marker
        marker_img = cv2.rectangle(marker_img, (0, height - marker_size), (marker_thickness_bottom, height),
                                   self.marker_color, -1)
        marker_img = cv2.rectangle(marker_img, (0, height - marker_thickness_bottom), (marker_size, height),
                                   self.marker_color, -1)

        # Bottom-right corner marker
        marker_img = cv2.rectangle(marker_img, (width - marker_thickness_bottom, height - marker_size), (width, height),
                                   self.marker_color, -1)
        marker_img = cv2.rectangle(marker_img, (width - marker_size, height - marker_thickness_bottom), (width, height),
                                   self.marker_color, -1)

        # Overlay the markers on the image
        result_img = img.copy()
        result_img = cv2.addWeighted(img, 0.5, marker_img, 1, 0)

        return result_img

    def print_image_with_markers(self, img):
        # Create an image with markers
        result_img = self.create_marker_image(img)

        # Save the image with markers to a file
        cv2.imwrite('printed_image.jpg', result_img)

        # Print the image using a printer
        # Add code here to send the image to a printer


class ImageCreator:
    def __init__(self, image_shape):
        self.image_shape = (2000, 2000)

    def create_speckle_image(self, dot_size=4, density=0.32, smoothness=2.0):
        image_shape = self.image_shape
        speckle_image = vlab.rosta_speckle(image_shape, dot_size=4, density=0.32, smoothness=2.0)
        downsampler = vlab.Downsampler(image_shape=image_shape, factor=4, fill=0.8, pixel_offset_stddev=0.1)
        downsampled_speckle = downsampler(speckle_image)
        return downsampled_speckle

    def apply_transformation(self, image, transformation):
        transformed_image = vlab.transform_image(image, transformation)
        return transformed_image

    def display_image(self, image, cmap='gray'):
        plt.imshow(image, cmap=cmap)
        plt.axis('off')
        plt.show()


class ImageRotator:
    def __init__(self, image):
        self.image = image

    def rotate(self, angle):
        rotated_image = imutils.rotate_bound(self.image, angle)
        return rotated_image


class ImageAligner:
    def __init__(self, image, rotated_image):
        self.rotated_image = rotated_image
        self.image1 = image

    def align_images(self):
        # Rotate the first image by 30 degrees clockwise using ImageRotator class
        # rotated_image = ImageRotator.rotate_image(self.image1, 30)

        # Convert images to grayscale
        gray_image1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(self.rotated_image, cv2.COLOR_BGR2GRAY)

        # Perform feature matching using SIFT
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray_image1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray_image2, None)

        # Create a BFMatcher object
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Extract keypoints for good matches
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate the affine transformation matrix
        transformation_matrix, _ = cv2.estimateAffinePartial2D(points2, points1)

        # Warp the second image to align with the first image
        aligned_image2 = cv2.warpAffine(self.rotated_image, transformation_matrix,
                                        (self.image1.shape[1], self.image1.shape[0]))

        return aligned_image2


class ImageComparer:
    #def mse(image1, image2):
    #    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    #    err /= float(image1.shape[0] * image1.shape[1])
    #    return err

    @staticmethod
    def mse(image1, image2):
        err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
        err /= float(image1.shape[0] * image1.shape[1])
        # Plot the images and the difference between them
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(image1, cmap='gray')
        plt.title('Image 1')
        plt.subplot(1, 3, 2)
        plt.imshow(image2, cmap='gray')
        plt.title('Image 2')
        plt.subplot(1, 3, 3)
        plt.imshow(image1 - image2, cmap='gray')
        plt.title('Difference')
        plt.tight_layout()
        plt.show()

        return err

    @staticmethod
    def calculate_residuals(img1, img2):
        # Convert images to grayscale for comparison
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Calculate the residuals by subtracting pixel values
        residuals = np.subtract(gray1, gray2)

        return residuals

    @staticmethod
    def calculate_least_squares_regression(residuals):
        # Calculate the least-squares regression by squaring the residuals
        regression = np.square(residuals)
        # Plot the regression values
        plt.plot(regression)
        plt.xlabel("Pixel Index")
        plt.ylabel("Regression Value")
        plt.title("Least-Squares Regression")
       # plt.show()
        return regression


def main():
    print("Please choose an option:")
    print("1. Digital tests")
    print("2. Real camera tests")
    # Specify the path to your image
    image_path = "img1.png"
    option = input("Enter your choice (1 or 2): ")
    # Set the image shape
    image_shape = (2000, 2000)

    # Create an instance of the ImageCreator class
    image_creator = ImageCreator(image_shape)

    # Create the speckle image for all the further test purposes
    speckle_image = image_creator.create_speckle_image(dot_size=10, density=0.5, smoothness=2.0)

    # Save the speckle image to the same path as the Python file
    speckle_image_path = os.path.join(os.path.dirname(__file__), "speckle_image.png")

    # Manually scale and convert the speckle image before saving it
    # cv2.normalize function is used to scale the speckle image to the range [0, 255], and the dtype parameter is set to cv2.CV_8U to ensure it is saved as an 8-bit image.
    # The scaled speckle image is then saved using cv2.imwrite,
    scaled_speckle_image = cv2.normalize(speckle_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(speckle_image_path, scaled_speckle_image)
    scaled_speckle_image_bgr = cv2.cvtColor(scaled_speckle_image, cv2.COLOR_GRAY2BGR)
    scaled_speckle_image_array = np.reshape(scaled_speckle_image_bgr,
                                            (scaled_speckle_image_bgr.shape[0], scaled_speckle_image_bgr.shape[1], 3))

    # Display the speckle image
    image_creator.display_image(scaled_speckle_image)
    if option == "1":
        # create an instance of ImageMarkerPrinter
        marker_printer = ImageMarkerPrinter()
        # create an image with markers for future orinting to perform the test
        marker_printer.print_image_with_markers(scaled_speckle_image_array)
        # Load the images
        # image = cv2.imread("image1.jpg")
        # image2 = cv2.imread("image2.jpg")

        # Create an instance of the ImageRotator class
        rotator = ImageRotator(scaled_speckle_image_array)

        # Rotate the image by 30 degrees
        rotated_image = rotator.rotate(30)

        # Create an instance of the ImageAligner class
        aligner = ImageAligner(scaled_speckle_image_array, rotated_image)

        # Align the rotated image with the original image
        aligned_image2 = aligner.align_images()
        image1 = scaled_speckle_image_array
        # Calculate the new dimensions to fit both the original and rotated image
        max_width = max(image1.shape[1], rotated_image.shape[1])
        max_height = max(image1.shape[0], rotated_image.shape[0])

        # Create a blank canvas with the maximum dimensions
        canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)

        # Calculate the position to place the original image on the canvas
        start_x1 = (max_width - image1.shape[1]) // 2
        start_y1 = (max_height - image1.shape[0]) // 2

        # Calculate the position to place the rotated image on the canvas
        start_x2 = (max_width - rotated_image.shape[1]) // 2
        start_y2 = (max_height - rotated_image.shape[0]) // 2

        # Embed the original image onto the canvas
        canvas[start_y1:start_y1 + image1.shape[0], start_x1:start_x1 + image1.shape[1]] = image1

        # Embed the rotated image onto the canvas
        canvas[start_y2:start_y2 + rotated_image.shape[0], start_x2:start_x2 + rotated_image.shape[1]] = rotated_image

        # Save the embedded image
        cv2.imwrite("embedded_image.png", canvas)

        # Save the aligned image
        cv2.imwrite("aligned_image.png", aligned_image2)

        # Calculate the MSE between the original image and the aligned image using ImageComparer class

        # mse_value = ImageComparer.mse(aligner.image1, aligned_image2)
        # print("Mean Square Error (MSE):", mse_value)

        # Calculate MSE
        mse_value = ImageComparer.mse(image1, aligned_image2)
        print("Mean Square Error (MSE):", mse_value)


        # Calculate residuals
        residuals = ImageComparer.calculate_residuals(image1, aligned_image2)
        print("Residuals:")
        print(residuals)

        # Calculate least-squares regression
        regression = ImageComparer.calculate_least_squares_regression(residuals)
        print("Least-Squares Regression:")
        print(regression)
    elif option == "2":
        # Example usage:
        image_path = 'printed_image_Test.jpg'
        # image = cv2.imread(image_path)
        image = cv2.imread(image_path)
        # to test MSE we have to provide different files to all images captured
        image_path2 = 'Lighting_from_left.jpg'
        image2 = cv2.imread(image_path2)
        # Specify the desired width and height for resizing
        width = 2000
        height = 2000
        # Resize the first image
        image = cv2.resize(image, (width, height))
       # cv2.imwrite("printed_image_resized.jpg", image)
        # Resize the second image
        image2 = cv2.resize(image2, (width, height))
       # cv2.imwrite("printed_image_resized.jpg", image2)

        scaled_speckle_image_array = image
        rotated_image = image2
        # Create an instance of the ImageAligner class
        aligner = ImageAligner(scaled_speckle_image_array, rotated_image)

        # Align the rotated image with the original image
        aligned_image2 = rotated_image#aligner.align_images()
        image1 = scaled_speckle_image_array

        # Save the aligned image
        # cv2.imwrite("aligned_image.png", aligned_image2)

        # Calculate the MSE between the original image and the aligned image using ImageComparer class

        # mse_value = ImageComparer.mse(aligner.image1, aligned_image2)
        # print("Mean Square Error (MSE):", mse_value)

        # Calculate MSE
        mse_value = ImageComparer.mse(image1, aligned_image2)
        print("Mean Square Error (MSE):", mse_value)

        # Calculate residuals
        residuals = ImageComparer.calculate_residuals(image1, aligned_image2)
        print("Residuals:")
        print(residuals)

        # Calculate least-squares regression
        regression = ImageComparer.calculate_least_squares_regression(residuals)
        print("Least-Squares Regression:")
        print(regression)


    else:
        print("You didnt choose the right option")


if __name__ == "__main__":
    main()
