import math
import numpy as np
import cv2


class ShapeRecognition(object):
    def __init__(self, img):
        self.img = img
        self.contours = None
        self.binary_img = None

    def get_binary_image(self, lower=[0, 0, 0], upper=[15, 15, 15]):
        self.binary_img = cv2.inRange(
            self.img, np.array(lower), np.array(upper))
        return self.binary_img

    def get_contours(self):
        (flags, self.contours, h) = cv2.findContours(
            cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return self.contours


if __name__ == "__main__":
    img = cv2.imread("shapes.jpg")
    shape_recognition = ShapeRecognition(img)

    binary_img = shape_recognition.get_binary_image()
    contours = shape_recognition.get_contours()

    combined = np.vstack((img, cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)))
    cv2.imshow("combined", combined)
    cv2.waitKey(0)
