# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


def display_four(images, names = ["First","Second","Third","Fourth"]):
    one, two, three, four = images
    one_name, two_name, three_name, four_name = names
    plt.figure(figsize=[20, 5])
    plt.subplot(141)
    plt.imshow(one, cmap='gray')
    plt.title(one_name)
    plt.subplot(142)
    plt.imshow(two, cmap='gray')
    plt.title(two_name)
    plt.subplot(143)
    plt.imshow(three, cmap='gray')
    plt.title(three_name)
    plt.subplot(144)
    plt.imshow(four, cmap='gray')
    plt.title(four_name)
    plt.show()
    cv2.waitKey(0)

def exercise_one(filename):
    # Load filename as a csv image. 0-flag reads as greyscale, 1 as BGR
    image_bgr = cv2.imread(filename)
    image_gs = cv2.imread(filename, 0)
    # Display an image
    # cv2.imshow("window", image_bgr)

    # NOTE OPENCV uses BGR not RGB!!! (reverse stack?) Quickly flip:
    # image_channel_reversed = image[:, :, ::-1]

    # split into bands
    b, g, r = cv2.split(image_bgr)

    # convert to other colour maps using static variables
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Use Matplot lib to show more complex figures
    # plt. kinda builds things iteratively until you switch to a new subplot
    # CHECK?: plt.imshow() adds the plot to the plt. plt.show() actually displays it
    img_merged = cv2.merge((b, g, r))
    display_four([r, g, b, img_merged[:, :, ::-1]], ["Red", "Green", "Blue", "Merged"])
    # plt.figure(figsize=[20, 5])
    # plt.subplot(141)
    # plt.imshow(r, cmap='gray')
    # plt.title("Red Channel")
    # plt.subplot(142)
    # plt.imshow(g, cmap='gray')
    # plt.title("Green Channel")
    # plt.subplot(143)
    # plt.imshow(b, cmap='gray')
    # plt.title("Blue Channel")
    # plt.subplot(144)
    # plt.imshow(img_merged[:, :, ::-1])
    # plt.title("Merged Output")
    # plt.show()
    # cv2.waitKey(0)


def exercise_two(filename): # Image Manipulation Exercise
    image_bgr = cv2.imread(filename, cv2.COLOR_BGR2RGB)

    # Remember, cv2 reads image as a matrix, so it's very easy to manipulate individual pixels
    image_bgr_copy = image_bgr.copy()
    image_bgr_copy[25:75, 25:75] = 255
    # cv2.imshow("window", image_bgr_copy)

    # You can also take subsets of images, like a matrix
    image_bgr_crop = image_bgr[200:300, 300:500]
    cv2.imshow("window", image_bgr_crop)

    # Or resize an image
    # fx/fy function are scale factors
    # resize1 = cv2.resize(image_bgr, None, fx=2, fy=2) # dsize is required, but you can make it None
    # Specify dimensions using dsize, interpolation choice
    dest_dimensions = [dest_width, dest_height] = [1000, 1000]
    resize1 = cv2.resize(image_bgr_crop, dest_dimensions, interpolation=cv2.INTER_AREA)
    resize2 = cv2.resize(image_bgr_crop, dest_dimensions, interpolation=cv2.INTER_NEAREST)
    resize3 = cv2.resize(image_bgr_crop, dest_dimensions, interpolation=cv2.INTER_LANCZOS4)

    # cv2.imshow("Window", resize2)
    display_four([image_bgr, resize1, resize2, resize3], ["Original", "INTER_AREA", "NN", "Lanczos"])
    # cv2.waitKey(0)

def exercise_three(filename): # Image Annotation
    image_bgr = cv2.imread(filename, cv2.COLOR_BGR2RGB)

    # Draw a line
    start, end = [(200, 100), (400, 100)]
    color = (0, 255, 255)
    anno_img = cv2.line(image_bgr, start, end, color, thickness=5, lineType=cv2.LINE_AA)
    plt.imshow(anno_img)
    plt.show()
    cv2.waitKey(0)
    # Also try...
    # cv2.rectangle()
    # cv2.putText()
    # cv2.circle()


def exercise_four(filename): # Image Enhancement
    image_rgb = cv2.imread(filename, cv2.COLOR_BGR2RGB)
    mod_matrix = np.ones(image_rgb.shape, dtype='uint8')

    # to adjust brightness, add or subtract a constant * matrix of the size of the image
    brighter = cv2.add(image_rgb, mod_matrix*50)
    darker = cv2.subtract(image_rgb, mod_matrix*50)

    # to modify the contrast, multiply the bands by a scalar
    # Note: need to cast to float to multiply, then recast back to a uint8.
    higher_contrast = np.uint8(np.clip(cv2.multiply(np.float64(image_rgb), mod_matrix * 1.3),0,255))
    lower_contrast = np.uint8(cv2.multiply(np.float64(image_rgb), mod_matrix * 0.7))
    # Further, higher contrast needs to be clipped to 255.
    higher_contrast = np.clip(higher_contrast, 0, 255)

    # display_four([brighter, darker, higher_contrast, lower_contrast], ["brighter", "darker", "higher_contrast", "lower_contrast]"])

    image_gray = cv2.imread(filename, 0)
    # Thresholding creates a binary mask from greyscale images. You can ignore the returned retval for now
    retval, image_top_thresh = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY)
    retval, image_low_thresh = cv2.threshold(image_gray, 170, 255, cv2.THRESH_BINARY)
    image_adp_thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)
    # Adaptive threshold helps to adjust the threshold as global trends occur throughout the image

    #display_four([image_gray, image_top_thresh, image_low_thresh, image_adp_thresh])

    # It may also be advantageous to do Bitwise operations
    result_and = cv2.bitwise_and(image_top_thresh, image_low_thresh, mask=None)
    result_or = cv2.bitwise_or(image_top_thresh, image_low_thresh, mask=None)
    result_xor = cv2.bitwise_xor(image_top_thresh, image_low_thresh, mask=None)
    display_four([image_gray, result_and, result_or, result_xor])

def exercise_five(): #Using A Camera Input
    s = 0
    if len(sys.argv) > 1:
        s = sys.argv[1]

    source = cv2.VideoCapture(s)

    win_name = "Camera Preview"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while cv2.waitKey(1) != 27: #escape key
        has_frame, frame = source.read()
        if not has_frame:
            break
        cv2.imshow(win_name, frame)

    source.release()
    cv2.destroyWindow(win_name)

def exercise_six():
    # Image Filtering (convolution)
    # Define modes
    names = ["Preview Mode", "Blurriung Filter", "Corner Feature Detector", "Canny Edge Detector"]
    PREVIEW, BLUR, FEATURES, CANNY = [0, 1, 2, 3]
    # PREVIEW - default, unfiltered view
    # CANNY - algorithm uses upper and lower threshold to determine if an edge or not
    # BLUR - uses a box filter to blur the image, the dimensions of the box are provided as input
    # FEATURES - uses feature_params to determine parameters that confirm a corner or not.

    # Extra parameters for corner detector
    feature_params = dict(maxCorners=500,
                          qualityLevel=0.2,
                          minDistance=15, #the book example, moving closer and farther away.
                          blockSize=9)
    s = 0
    if len(sys.argv)>1:
        s = sys.argv[1]

    image_filter = PREVIEW
    alive = True
    win_name = "Camera Filters"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    result = None

    source = cv2.VideoCapture(s)

    while alive:
        has_frame, frame = source.read()
        if not has_frame:
            break

        frame = cv2.flip(frame, 1) #flips frame horizontally

        if image_filter == PREVIEW:
            result = frame
        elif image_filter == CANNY:
            result = cv2.Canny(frame, 145, 150)
        elif image_filter == BLUR:
            result = cv2.blur(frame, (10, 10))
        elif image_filter == FEATURES:
            result = frame
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
            if corners is not None:
                for x, y in np.float32(corners).reshape(-1, 2):
                    # Not in video, but the X, Y need to be cast back to INT or else circle() will be mad
                    cv2.circle(result, (int(x), int(y)), 10, (0, 255, 0), 1)

        cv2.imshow(win_name, result)
        key = cv2.waitKey(1)
        if key == ord('Q') or key == ord('q') or key == 27:
            alive = False
        elif key == ord('C') or key == ord('c'):
            image_filter = CANNY
        elif key == ord('B') or key == ord('b'):
            image_filter = BLUR
        elif key == ord('F') or key == ord('f'):
            image_filter = FEATURES
        elif key == ord('P') or key == ord('p'):
            image_filter = PREVIEW

    source.release()
    cv2.destroyWindow(win_name)
    ### BIG PRO HINT
    # Recommended to do blurring/smoothing as a preprocessing step to reduce the amount
    # of overall noise in other processing steps, such as numerical gradients/slopes
    # Especially feature extractions.


if __name__ == '__main__':
    filename = 'testimage.png'
    # exercise_one(filename)
    # exercise_two(filename)
    # exercise_three(filename)
    # exercise_four(filename)
    # exercise_five()
    exercise_six()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
