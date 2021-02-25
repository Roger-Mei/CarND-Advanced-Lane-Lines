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

# Read in an image from 'test_images'
img = mpimg.imread('test_images/test1.jpg')

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

    # create a binary copy
    dir_binary = np.zeros_like(absgraddir)

    # Apply the threshold
    dir_binary[(absgraddir > dir_thresh[0]) & (absgraddir < dir_thresh[1])] = 1
    return dir_binary

# Convert to HLS color space and separate S channel
hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
s_channel = hls[:, :, 2]

# Transform the original image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)




# Apply a perspective transform to rectify binary image ("birds-eye view").
# Detect lane pixels and fit to find the lane boundary.
# Determine the curvature of the lane and vehicle position with respect to center.
# Warp the detected lane boundaries back onto the original image.
# Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


# ---------------------------------------------Code for writing ouput file------------------------------------------------#
f, (ax1, ax2) = plt.subplots(1,2, figsize = (24,9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize = 50)
ax2.imshow(undist)
ax2.set_title('Undistorted Image', fontsize = 50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()