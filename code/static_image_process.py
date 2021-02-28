# ----------------------------------------------- Import environment ----------------------------------------------------#
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob

import src
from src import functions

# ------------------------------------------------ Execution part --------------------------------------------------------#

###########################################################################################################################
# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.                     #
###########################################################################################################################

# Read in and make a list of calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

# Create arrays to store object points and image points from all the images
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# Prepare object points 
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1,2) # x, y coordinates

for fname in images:
    # Read in each image
    img = mpimg.imread(fname)

    # Convert image into gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Find the chessboard corner
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If corners are found, add object points and image points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

# Camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

###########################################################################################################################
# Apply a distortion correction to raw images.                                                                            #
###########################################################################################################################
"""
Implementation of this section
"""
# Read in an image from 'test_images'
img = mpimg.imread('../test_images/test1.jpg')

# undistort the raw image
undist = src.functions.cal_undistort(img, objpoints, imgpoints)

###########################################################################################################################
# Use color transforms, gradients, etc., to create a thresholded binary image.                                            #
###########################################################################################################################
"""
Implementation of this section
"""
combined_binary = src.functions.pipeline_threshlold(undist, sx_thresh=(30,100))

###########################################################################################################################
# Apply a perspective transform to rectify binary image ("birds-eye view").                                               #
###########################################################################################################################
"""
Implementation of this section
"""
warped_binary, Minv = src.functions.warp(combined_binary)

###########################################################################################################################
# Detect lane pixels and fit to find the lane boundary.                                                                   #
###########################################################################################################################
"""
Implementation of this section
"""
tmp_result, left_fit, right_fit, ploty = src.functions.fit_polynomial(warped_binary)

###########################################################################################################################
# Determine the curvature of the lane and vehicle position with respect to center.                                        #
###########################################################################################################################
"""
Implementation of this section
"""
left_curverad, right_curverad = src.functions.measure_curvature_real(left_fit, right_fit, warped_binary)
center_dist = src.functions.measure_center_dist_real(warped_binary, left_fit, right_fit)

###########################################################################################################################
# Warp the detected lane boundaries back onto the original image.                                                         #
###########################################################################################################################
"""
Implementation of this section
"""
unwarped_img = src.functions.draw_lane(undist, warped_binary, left_fit, right_fit, ploty, Minv)

###########################################################################################################################
# Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.           #
###########################################################################################################################
"""
Implementation of this section
"""
unwarped_img_with_data = src.functions.add_numerical_estimation(unwarped_img, left_curverad, right_curverad, center_dist)

###########################################################################################################################
# Build up the image process pipeline                                                                                     #
###########################################################################################################################
"""
Implementation of this section
"""
result = src.functions.process_image(img)

plt.imshow(result)
plt.show()

# ---------------------------------------------Code for writing ouput file------------------------------------------------#
# f, (ax1, ax2) = plt.subplots(1,2, figsize = (24,9))
# f.tight_layout()
# ax1.imshow(warped_binary)
# ax1.set_title('Warped Binary', fontsize = 20)
# ax2.imshow(warped2)
# ax2.set_title('Warped Image', fontsize = 20)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()