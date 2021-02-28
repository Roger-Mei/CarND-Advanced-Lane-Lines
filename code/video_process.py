# ----------------------------------------------- Import environment ----------------------------------------------------#
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob

import src
from src import functions
from src import line

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
# Video processing                                                                                                        #
###########################################################################################################################
"""
Useful function construction for this part
"""
def video_process(img):
    # NOTE: before running the pipeline, you should have the objpoints and imgpoints ready
    # Undistort image
    undist = src.functions.cal_undistort(img, objpoints, imgpoints)

    # Implement the threshold pipeline to get the 
    combined_binary = src.functions.pipeline_threshlold(undist, sx_thresh=(30,100))

    # Warp the binary image via perspective transform
    warped_binary, Minv = src.functions.warp(combined_binary)

    # if both left and right lines were detected last frame, use polyfit_using_prev_fit, otherwise use fit_polynomial
    if not left_line.detected or not right_line.detected:
        result, left_fit, right_fit, ploty = src.functions.fit_polynomial(warped_binary)
    else:
        left_fit, right_fit, ploty = src.functions.polyfit_using_prev_fit(warped_binary, left_line.current_fit, right_line.current_fit)

    # Estimate the curvature radius as well as the distance to lane center
    left_curverad, right_curverad = src.functions.measure_curvature_real(left_fit, right_fit, warped_binary)

    # Update Line info
    left_line.add_fit(left_fit, ploty, left_curverad)
    right_line.add_fit(right_fit, ploty, right_curverad)

    # Recalculate car's distance to center
    center_dist = src.functions.measure_center_dist_real(warped_binary, left_line.current_fit, right_line.current_fit)

    # Warp the detected lane boundaries back onto the original image
    unwarped_img = src.functions.draw_lane(undist, warped_binary, left_line.current_fit, right_line.current_fit, ploty, Minv)

    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
    unwarped_img_with_data = src.functions.add_numerical_estimation(unwarped_img, left_line.radius_of_curvature, right_line.radius_of_curvature, center_dist)

    return unwarped_img_with_data



"""
Implementation of this section
"""
# Initialize two lines
left_line = src.line.Line()
right_line = src.line.Line()

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

white_output = '../test_videos_output/solidWhiteRight2.mp4'
clip1 = VideoFileClip("../project_video.mp4")
white_clip = clip1.fl_image(video_process) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)