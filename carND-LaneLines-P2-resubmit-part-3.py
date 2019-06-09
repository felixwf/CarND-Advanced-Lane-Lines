import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import glob

# Define a class to receive the characteristics of each line detection


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


def fit_poly(img_shape, leftx, lefty, rightx, righty):
                                                                                                                                                                                                                                                                 ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fit, right_fit, left_fitx, right_fitx, ploty


def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/700  # meters per pixel in x dimension

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                                                                         left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                                                                           right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(
        binary_warped.shape, leftx, lefty, rightx, righty)
    # Fit new polynomials (in the real world)
    left_fit_cr, right_fit_cr, left_fitx_cr, right_fitx_cr, ploty_cr = fit_poly(
        binary_warped.shape, leftx * xm_per_pix, lefty * ym_per_pix, rightx * xm_per_pix, righty * ym_per_pix)

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = (
                    (1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = (
        (1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    # Calculation of R_curve (radius of curvature in the real world)
    left_curverad_real = (
        (1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad_real = (
        (1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array(
        [np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array(
        [np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##

    return result, left_curverad, right_curverad, left_curverad_real, right_curverad_real, left_fit, right_fit


def binary_warped_generate(image):
    c_b, thresholded_binary = pipeline(
        image, s_thresh=(90, 255), sx_thresh=(20, 100))
    warped_thresholded_binary = warp(thresholded_binary)
    return warped_thresholded_binary


def warp(img):
    img_size = (img.shape[1], img.shape[0])

    # Four source coordinate
    src = np.float32([[593, 450], [690, 450], [200, 720], [1120, 720]])

    # Four desired coordinates
    dst = np.float32([[200, 0],	[1000, 0], [200, 720], [1000, 720]])

    # Compute the perspective transform, M
    M = cv2.getPerspectiveTransform(src, dst)

    # Could compute the inverse also by swapping the input parameters
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Create warped image - use linear interpolation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, Minv


def hls_thresh(img, channel='s', thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel == 'h':
        channel = hls[:, :, 0]
    elif channel == 'l':
        channel = hls[:, :, 1]
    else:
        channel = hls[:, :, 2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output


def lab_thresh(img, channel='l', thresh=(0, 255)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    if channel == 'l':
        channel = lab[:, :, 0]
    elif channel == 'a':
        channel = lab[:, :, 1]
    else:
        channel = lab[:, :, 2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output


def luv_thresh(img, channel='l', thresh=(0, 255)):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if channel == 'l':
        channel = luv[:, :, 0]
    elif channel == 'u':
        channel = luv[:, :, 1]
    else:
        channel = luv[:, :, 2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output


def rgb_thresh(img, channel='r', thresh=(0, 255)):
    rgb = img
    if channel == 'r':
        channel = rgb[:, :, 0]
    elif channel == 'g':
        channel = rgb[:, :, 1]
    else:
        channel = rgb[:, :, 2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output


# Edit this function to create your own pipeline.
def pipeline(img, lab_b_thresh=(145, 200), luv_l_thresh=(215, 255)):
    img = np.copy(img)
    binary_output_1 = lab_thresh(img, channel='b', thresh=lab_b_thresh)
    binary_output_2 = luv_thresh(img, channel='l', thresh=luv_l_thresh)

    binary_output = np.zeros_like(binary_output_1)
    binary_output[(binary_output_1 == 1) | (binary_output_2 == 1)] = 1

    # Stack each channel
    color_binary = np.dstack(
        (binary_output_1, binary_output_2, np.zeros_like(binary_output_2))) * 255

    return color_binary, binary_output


def find_lane_pixels(binary_warped):
    '''
    在二进制，经过Transform的图片中，寻找左右两条车道线的所有的pixels
    返回所有的pixels，返回带有滑动窗口的图片
    '''
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    '''
    在二进制图片中，拟合出两条车道线的位置。并且用两条黄色的线绘制在图片中
    '''
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit, right_fit


def calibration_params_cal():
    '''
    用'camera_cal'中的图片，校准相机，得到校准相机用的坐标点
    '''
    # Read in adn make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')

    # Arrays to store object points and image points from all the images
    objpoints = []
    imgpoints = []

    # Prepare object points
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    for fname in images:
        # Read in each image
        img = mpimg.imread(fname)

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If corners are found, add object points, image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img.shape[1:], None, None)

    return ret, mtx, dist, rvecs, tvecs


def undistort_image(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def image_process(input_image, output_image, mtx, dist):
    left_fit = Line()
    right_fit = Line()

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/700  # meters per pixel in x dimension

    left_fit.current_fit = np.array([0.0, 0.0, 300.0])
    right_fit.current_fit = np.array([0.0, 0.0, 900.0])

    image = mpimg.imread(input_image)
    # input_clip = VideoFileClip(input_video).set_duration(3.0)

    def process_image(image):

        # Step 1: Undistort each frame
        img = undistort_image(image, mtx, dist)

        # Step 2: Convert each frame into a binary image, which can display lane lines clearly,
        # using the HLS and gradient threshold together

        color_binary, thresholded_binary = pipeline(image, lab_b_thresh=(145, 200), luv_l_thresh=(215, 255))

        # Step 3: Convert the image into a top-down view for calculate the curve and fitpoly
        warped_thresholded_binary, M, Minv = warp(thresholded_binary)

        # Step 4: Find the lane lines
        # out_img, left_fit.current_fit, right_fit.current_fit = fit_polynomial(warped_thresholded_binary)
        result, left_curverad, right_curverad, left_curverad_real, right_curverad_real, left_fit.current_fit, right_fit.current_fit = search_around_poly(
            warped_thresholded_binary, left_fit=left_fit.current_fit, right_fit=right_fit.current_fit)

        left_fit.radius_of_curvature = left_curverad_real
        right_fit.radius_of_curvature = right_curverad_real

        # Step 5: Display the lane lines on the image
        yMax = img.shape[0]
        ploty = np.linspace(0, yMax - 1, yMax)
        color_warp = np.zeros_like(img).astype(np.uint8)

        # Step 5.1: Calculate points.
        left_fitx = left_fit.current_fit[0]*ploty**2 + \
            left_fit.current_fit[1]*ploty + left_fit.current_fit[2]
        right_fitx = right_fit.current_fit[0]*ploty**2 + \
            right_fit.current_fit[1]*ploty + right_fit.current_fit[2]

        # Step 5.2: Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Step 5.3: Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Step 5.4: Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(
            color_warp, Minv, (img.shape[1], img.shape[0]))
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        # Step 6: Display the lane lines info on the image
        left_curve_base = left_fit.current_fit[0] * image.shape[0] ** 2 + \
            left_fit.current_fit[1] * image.shape[0] + left_fit.current_fit[2]
        right_curve_base = right_fit.current_fit[0] * image.shape[0] ** 2 + \
            right_fit.current_fit[1] * \
            image.shape[0] + right_fit.current_fit[2]
        car_offset = ((left_curve_base + right_curve_base) /
                      2 - image.shape[1] / 2) * xm_per_pix
        text1 = "Curvature radius (left, right) = (" + str(
            left_fit.radius_of_curvature) + "m, " + str(right_fit.radius_of_curvature) + "m)"
        text2 = "Car offset = " + str(car_offset) + "m"
        cv2.putText(result, text1, (40, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        cv2.putText(result, text2, (40, 100),
                    cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

        # Step 7: Return the processed image
        return result

    
    result = process_image(image)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image, result)

if __name__ == "__main__":

    # Step 1: Calibrate the camera, get the calibration params
    ret, mtx, dist, rvecs, tvecs = calibration_params_cal()

    # Step 2: Processing the video frame by frame
    input_image = './test_images/test2.jpg'
    output_image = './output_images/lane_area_add_on_original_image.jpg'
    image_process(input_image, output_image, mtx, dist)
    # # Read an image
