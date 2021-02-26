# ----------------------------------------------- Import environment ----------------------------------------------------#
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob

# ------------------------------------------------ Execution part --------------------------------------------------------#

###########################################################################################################################
# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.                     #
###########################################################################################################################

# Read in and make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

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
Useful function construction for this part
"""
def cal_undistort(img, objpoints, imgpoints):
    # Transform image into gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Find the corners of the borad 
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # Camera calibration and undistort
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

"""
Implementation of this section
"""
# Read in an image from 'test_images'
img = mpimg.imread('test_images/test2.jpg')

# undistort the raw image
undist = cal_undistort(img, objpoints, imgpoints)

###########################################################################################################################
# Use color transforms, gradients, etc., to create a thresholded binary image.                                            #
###########################################################################################################################

"""
Useful function construction for this part
"""
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply x or y gradient and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Create a binary copy of scaled_sobel
    grad_binary = np.zeros_like(scaled_sobel)

    # Apply the threshold
    grad_binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1

    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take both sobel x and y grdients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)

    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)

    # Rescale back to 8 bit integer
    scaled_mag = np.uint8(255*gradmag/np.max(gradmag))

    # create a binary copy
    mag_binary = np.zeros_like(scaled_mag)

    # Apply the threshold
    mag_binary[(scaled_mag > mag_thresh[0]) & (scaled_mag < mag_thresh[1])] = 1

    return mag_binary

def dir_threshold(img, sobel_kernel=3, dir_thresh=(0, np.pi/2)):
    # Convert image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take both sobel x and y grdients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)

    # Take the absolute value of the gradient direction
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # Create a binary copy
    dir_binary = np.zeros_like(absgraddir)

    # Apply the threshold
    dir_binary[(absgraddir > dir_thresh[0]) & (absgraddir < dir_thresh[1])] = 1
    return dir_binary

def hls_select(img, thresh=(0, 255)):
    # Convert color to HLS scale
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    # Select the S channel
    s_channel = hls[:,:,2]

    # Create a binary copy
    hls_binary = np.zeros_like(s_channel)

    # Apply the threshold
    hls_binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return hls_binary

def pipeline(undist, s_thresh=(170, 255), sx_thresh=(20, 100)):
    # Apply HLS color space threshold
    s_binary = hls_select(undist, thresh=s_thresh)

    # Apply sobel X gradient threshold
    sx_binary = abs_sobel_thresh(undist, orient='x', sobel_kernel=5, thresh=sx_thresh)

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(s_binary), sx_binary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[(s_binary == 1) | (sx_binary == 1)] = 1
    return combined_binary

"""
Implementation of this section
"""
combined_binary = pipeline(undist, sx_thresh=(30,100))

###########################################################################################################################
# Apply a perspective transform to rectify binary image ("birds-eye view").                                               #
###########################################################################################################################

"""
Useful function construction for this part
"""
def warp(img):
    # Retrive image size
    img_size = (img.shape[1], img.shape[0])

    # Four source coordinates
    src = np.float32(
        [[760, 480],
         [1110, 690],
         [290, 690],
         [570, 480]])

    # Four desired coordinates
    dst = np.float32(
        [[920, 0],
         [920, 720],
         [280, 720],
         [280, 0]])

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Create warped image - use linear interpolation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped

"""
Implementation of this section
"""
warped_binary = warp(combined_binary)

###########################################################################################################################
# Detect lane pixels and fit to find the lane boundary.                                                                   #
###########################################################################################################################

"""
Useful function construction for this part
"""
def hist(img):
    # Grab only the bottom half of the image
    bottom_half = img[img.shape[0]//2:, :]

    # Sum across image pixels vertically
    histgram = np.sum(bottom_half, axis=0)

    return histgram


def find_lane_pixels(warped_binary):
    # Create histogram of image binary activations
    histogram = hist(warped_binary)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((warped_binary, warped_binary, warped_binary))*255

    # Find the peaks of the left and right half of the histograms
    midpoint = np.int(histogram.shape[0]//2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows
    window_height = np.int(warped_binary.shape[0]//nwindows)

    # identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = left_base
    rightx_current = right_base

    # Create empty list to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Loop through the windows
    for window in range(nwindows):
        # Identify window bounderies in x and y, as well as right and left
        win_y_low = warped_binary.shape[0] - (window + 1)*window_height
        win_y_high = warped_binary.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy > win_y_low) & (nonzeroy < win_y_high) 
        & (nonzerox > win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy > win_y_low) & (nonzeroy < win_y_high) 
        & (nonzerox > win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Justify the need of recentering windows
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds])) 
        
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoid an error if the above is not implemented fully
        pass

    # Extract the left line and right line pixels
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(warped_binary):
    # Find out lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped_binary)
    
    # Find second order polynomial to each line
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if 'left_fit' or 'right_fit' are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color = 'yellow')
    plt.plot(right_fitx, ploty, color = 'yellow')

    return out_img

"""
Implementation of this section
"""

out_img = fit_polynomial(warped_binary)

plt.imshow(out_img)
plt.show()


# Determine the curvature of the lane and vehicle position with respect to center.
# Warp the detected lane boundaries back onto the original image.
# Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


# ---------------------------------------------Code for writing ouput file------------------------------------------------#
# f, (ax1, ax2) = plt.subplots(1,2, figsize = (24,9))
# f.tight_layout()
# ax1.imshow(warped_binary)
# ax1.set_title('Warped Binary', fontsize = 20)
# ax2.imshow(warped2)
# ax2.set_title('Warped Image', fontsize = 20)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()