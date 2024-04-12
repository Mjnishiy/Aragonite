# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    plt.figure(figsize=[20, 5])
    plt.subplot(141)
    plt.imshow(r, cmap='gray')
    plt.title("Red Channel")
    plt.subplot(142)
    plt.imshow(g, cmap='gray')
    plt.title("Green Channel")
    plt.subplot(143)
    plt.imshow(b, cmap='gray')
    plt.title("Blue Channel")
    img_merged = cv2.merge((b, g, r))
    plt.subplot(144)
    plt.imshow(img_merged[:, :, ::-1])
    plt.title("Merged Output")
    plt.show()
    cv2.waitKey(0)


def exercise_two(filename):
    # Image Manipulation Exercise
    image_bgr = cv2.imread(filename)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filename = 'testimage.png'
    # exercise_one(filename)
    exercise_two(filename)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
