# Udacity Self Driving Car Nanondegree

## Project 4: Advanced Lane Finding

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Project Structure

This project is self-contained in the single Jupyter Notebook file, [LaneFinding.ipynb](LaneFinding.ipynb).

## Final results

<iframe width="560" height="315" src="https://www.youtube.com/embed/IRSBUBQPhng?ecver=1" frameborder="0" allowfullscreen></iframe>

<a href="https://youtu.be/IRSBUBQPhng" target="_blank">
  <img src="http://img.youtube.com/vi/IRSBUBQPhng/0.jpg" border="10" />
</a>

## Camera Calibration

Camera calibration was performed using the standard chessboard images which were provided in
[https://github.com/udacity/CarND-Advanced-Lane-Lines/tree/master/camera_cal](the Udacity Github
repo) for this project.


Stepping through each image I used the `cv2.findChessboardCorners` method which can be found in the
second cell of the Notebook. If corners were found for a particular images I kept track of the
object points corresponding image points.  In this case `objp` (object points) are the `x, y, z`
coordinates of the image which don't change from image to images (since we assume all images are of
the same size).  `imgpoints` is a list/array of `x, y` coordinates for the image corners which will
vary from image to image and corner to corner.  Both of these lists of coordinates are saved as
global variables and used to undistort camera images from the car.

To actually perform the undistortion I created a small helper function named `cal_undistort` in the
first cell. This is really a light wrapper around the `cv2.calibrateCamera()` and `cv2.undistort()` functions.

Below is a single example image with an original image (with the found corners overlayed) and
resulting undistorted image.

![](example-images/chessboard-calibration.png)

As a sanity check I applied the undistortion to one of the sample images from the freeway and
obtained the following which looks accurate:

![](example-images/road-test-calibration.png)


## Pipeline (single images)

### Thresholding using color transformations and gradients

The next step was creating a pipeline which takes an undistorted image as input and returns a
binary/thresholded image as output.  This code can be seen in the `pipeline` command.

The thresholding consists of three steps:

- Convert the image to `HLS` color space and extract the `S` channel. The `S` channel was chosen
  because it does the best job of clearly identifying lanes
- Create a binary image where pixels are kept if they are within my thresholded values of `(95,
  255]`
- Create an other image by applying a Sobel filter in the x direction, making sure to normalize the
  results
- Again, create a binary image where pixels are kept if they are within my thresholded values of
  `[20, 100]`. These values seemed to keep the features I was interested in and kept a minimal
  amount of noise.
- Finally, create a final binary image which is effectively a union of the two images where pixels
  are "on" for either one of the two images.

An example is shown below:

![](example-images/pipeline.png)


### Creating bird's eye view of lanes

Code for this can be seen in the notebook "Apply a perspective transform..."

To determine which starting points to use for the creation of a bird's eye view I eyeballed a few
points on the image:

```
src = np.float32([
    [265, 685], # bottom left
    [586, 455], # top left
    [698, 455], # top right
    [1055, 685], # bottom right
])
```

Placing these points on the image itself looked like this...you can see four dots on the lane
lines: two on the bottom of the image and two near the center of the image.

!()[example-images/transform-start-points.png]

After some trial and error, I came to a final set of destination points which produced the final
image below.

```
dst = np.float32([
    [255, h - 10], # bottom left
    [245, 0], # top left
    [1065, 0], # top right
    [1055, h - 10], # bottom right
])
```

![](example-images/transform-binary-image.png)


### Fitting a polynomial

Using the example code that I worked with in the project I did the work of fitting and visualizing
a polynomial on the lane lines.  This work and code is wrapped up inside the final
`LaneLineDetector` class for clarity.  Below is an image which I produced during the exploration
phase to ensure that the code was working as expected:

![](example-images/polyfit-with-boxes.png)

The code which does this work can be found in `LaneLineDetector.init_polyfit`, _without_ the code
that actually drew the bounding boxes and polyfit lines since that is extraneous for the final
submission.

The algorithm to detect the lane shape and fit a polynomial consists of:

- Taking a histogram of the bottom half of the image and find the "peaks" for both left and right
  sides. The assumption here is that the "on" pixels will be clumped together representing the lane
  lines
- Using a pre-defined bounding box side and number of iterations, walk up the image and recenter
  the windows on their respective new "peaks"...taking the mean `x` value as the center of the
  lane/bounding box. Save the `x, y` indices for all of the "on" pixels in these windows. This 
- Using all of the "on" `x, y` indices get the corresponding coordinates and fit a second-order
  polynomial
- At each iteration, I save the best fit and corresponding constants for the polynomial

After this is done for the first time, subsequent iterations will take shorter approach since the
sliding window method had already found the starting point for the lanes lines.  In
`LaneLineDetector.find_next_polyfit` the work of finding the non-zero pixels is done again but
without the sliding window process and _with_ a small margin.

### Calculating radius of curvature

The radius is calculated in `LaneLineDetector.calculate_radius`. Given a an image I fit a new
polynomial in real-world space where "real-world" really means "meters per pixel". These are
calculated based on the standard size of a highway lane:

```
ym_per_pix = 30 / 720 # meters per pixel in y dimension
xm_per_pix = 3.7 / 700 # meters per pixel in x dimension
```

The polyfit works because we already have the `x, y` coordinates of the lane, so this step is
simply a matter of doing the same math with a different scale.

After the polyfit occurs and we have our new constants we can use them to calculate curvature using
the final equation discussed in the project description. Python implementation for that equation,
where `self._y_eval` is just `height - 1` and `l1, l2, l3` are the constant terms from our
polynomial:

```
left_curverad =  ((1 + (2 * l1 * self._y_eval * ym_per_pix + l2)**2)**1.5) / np.absolute(2 * l1)
```


### Calculating offset from center

I calculated the car's position with respect to center in the
`LaneLineDetector.calculate_car_position` method.  The process here is quite simple since we
already have an equation to calculate the `x` location for a given `y` input.  Here, `y` is the
bottom of the image or the actual height of the image.

- Calculate the left and right `x` positions for max `y` value using our already-calculated
  polynomial. This is the bottom of the image and
  denotes the left and right boundary of the lanes immediately in front of the car.
- Determine the lane width by taking a difference between the right and left `x` locations
- Calculate the lane center by adding `lane_width / 2` to the left `x` position
- If lane center is greater than `image_width / 2` the car is to the left of center, and vice
  versa.
- Finally apply the same "meters per pixel" conversion to the pixel offset from center


### Drawing lane area on image

In `LaneLineDetector.draw_region` I create a new image to draw the lane line fits and shape. Much
of this is `numpy` array wrangling to build up the new image.

Towards the end of this function I use the `cv2.warpPerspective` function again but this time apply
an inverse transformation.  This takes us from the bird's eye view perspective back into the
"normal" perspective which is simply the original.


## Discussion

Overall I think this code performed fairly well.  There was a bit of jitter in the calculation for
the radius which makes me think the images/video look better than the calculations themselves.
There must be some error in the calculations because the radius values are changing much too
quickly during the move. 

To improve this I would spend more time playing with the gradient and thresholding numbers to find
an optimum pipeline to use.  Additionally, I would look at using different types of filtering and
color spaces.
