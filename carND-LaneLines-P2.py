import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip


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
	left_fit_cr, right_fit_cr, left_fitx_cr, right_fitx_cr, ploty_cr = fit_poly(binary_warped.shape,
						leftx * xm_per_pix,
						lefty * ym_per_pix,
						rightx * xm_per_pix,
						righty * ym_per_pix)

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

	return result, left_curverad, right_curverad, left_curverad_real, right_curverad_real


def binary_warped_generate(image):
	c_b, thresholded_binary = pipeline(
		image, s_thresh=(90, 255), sx_thresh=(20, 100))
	warped_thresholded_binary = warp(thresholded_binary)
	return warped_thresholded_binary


def warp(img):
	img_size = (img.shape[1], img.shape[0])

	# Four source coordinate
	src = np.float32([
		[593, 450],
		[690, 450],
		[200, 720],
		[1120, 720]
	])

	# Four desired coordinates
	dst = np.float32([
		[300, 0],
		[900, 0],
		[300, 720],
		[900, 720]
	])

	# Compute the perspective transform, M
	M = cv2.getPerspectiveTransform(src, dst)

	# Could compute the inverse also by swapping the input parameters
	Minv = cv2.getPerspectiveTransform(src, dst)

	# Create warped image - use linear interpolation
	warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

	return warped


# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
	img = np.copy(img)
	# Convert to HLS color space and separate the V channel
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	l_channel = hls[:, :, 1]
	s_channel = hls[:, :, 2]
	# Sobel x
	sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
	# Absolute x derivative to accentuate lines away from horizontal
	abs_sobelx = np.absolute(sobelx)
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

	# Threshold x gradient
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) &
			  (scaled_sobel <= sx_thresh[1])] = 1

	# Threshold color channel
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
	# Stack each channel
	color_binary = np.dstack(
		(np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
	combined_binary = np.zeros_like(sxbinary)
	combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
	return color_binary, combined_binary



def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
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
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
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
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
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

    return out_img


def process_image(image):
	# Step 1: Convert into binary image with perspective transform
	color_binary, thresholded_binary = pipeline(
		image, s_thresh=(90, 255), sx_thresh=(20, 100))
	warped_thresholded_binary = warp(thresholded_binary)
	

	# Step 2: Blur


	# Step 3: Canny


	# Step 4: Cover


	# Step 5: Hough

	
	# Step 6: Add on the original image
	result = warped_thresholded_binary

	return result



if __name__ == "__main__":
	# white_output = 'project_video_output.mp4'
	# clip1 = VideoFileClip("project_video.mp4")
	# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
	# white_clip.write_videofile(white_output, audio=False)
	
	# Read an image
	image = mpimg.imread('./test_images/straight_lines1.jpg')
	# image = mpimg.imread('./test_images/test1.jpg')

	result = process_image(image)

	# Plot the result
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	f.tight_layout()

	ax1.imshow(image)
	ax1.set_title('Original Image', fontsize=40)

	ax2.imshow(result)
	ax2.set_title('color_binary Result', fontsize=40)

	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()
