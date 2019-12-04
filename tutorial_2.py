import cv2
import numpy as np
import math


class ShapeRecognition(object):
    def __init__(self, img):
        self.img = img
        self.debug_img = img
        self.contours = None
        self.binary_img = None

    def create_binary_img(self, lower=[0, 0, 0], upper=[15, 15, 15]):
        self.binary_img = cv2.inRange(
            self.img, np.array(lower), np.array(upper))
        kernel = np.ones((5, 5), np.uint8)
        self.binary_img = cv2.morphologyEx(
            self.binary_img, cv2.MORPH_CLOSE, kernel)
        self.binary_img = cv2.dilate(self.binary_img, kernel, iterations=1)
        return self.binary_img

    def find_contours(self):
        (flags, self.contours, h) = cv2.findContours(
            self.binary_img,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return self.contours

    def get_angle(self, p1, p2, p3):

        vector1 = [(p1[0][0] - p2[0][0]), (p1[0][1] - p2[0][1])]
        vector2 = [(p1[0][0] - p3[0][0]), (p1[0][1] - p3[0][1])]

        cos = [((vector1[0] * vector2[0]) + (vector1[1] * vector2[1]))/((math.sqrt(math.pow(vector1[0], 2)
                                                                                   )+math.pow(vector1[1], 2)) + (math.sqrt(math.pow(vector2[0], 2))+math.pow(vector2[1], 2)))]
        degree = math.degrees(cos[0])
        degree = 90 - degree

        return degree

    def draw_debug_text(self, contour, cx, cy, text):
        cv2.drawContours(self.debug_img, [contour], 0, (0, 0, 255), 3)
        cv2.putText(self.debug_img, text, (cx-35, cy+65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2, cv2.LINE_AA)

    def find_shapes(self):
        self.create_binary_img()
        self.find_contours()
        for n, cnt in enumerate(self.contours):
            # Number of curves in the contour
            approx = cv2.approxPolyDP(
                cnt, 0.02*cv2.arcLength(cnt, True), True)
            M = cv2.moments(cnt)
            if M['m00'] > 300:
                cx = int(M['m10']/M['m00'])  # x-position of center
                cy = int(M['m01']/M['m00'])  # y-position of center
                p1 = approx[0]
                p2 = approx[1]
                p3 = approx[-1]
                if len(approx) >= 3 and len(approx) <= 3:  # for a triangle
                    self.draw_debug_text(cnt, cx, cy, "Triangle")


if __name__ == "__main__":
    img = cv2.imread("shapes.jpg")
    shape_recognition = ShapeRecognition(img)

    shape_recognition.find_shapes()

    combined = np.vstack(
        (shape_recognition.debug_img, cv2.cvtColor(shape_recognition.binary_img, cv2.COLOR_GRAY2BGR)))
    cv2.imshow("combined", combined)
    cv2.waitKey(0)