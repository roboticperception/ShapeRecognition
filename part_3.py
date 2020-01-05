import math
import numpy as np
import cv2


class ShapeRecognition(object):
    def __init__(self, img):
        self.img = img
        self.contours = None
        self.binary_img = None
        self.debug_img = img.copy()

    def get_binary_image(self, lower=[0, 0, 0], upper=[15, 15, 15]):
        self.binary_img = cv2.inRange(
            self.img, np.array(lower), np.array(upper))
        kernel = np.ones((5, 5), np.uint8)
        self.binary_img = cv2.dilate(self.binary_img, kernel, iterations=1)
        return self.binary_img

    def get_contours(self):
        self.contours, h = cv2.findContours(
            self.binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        return self.contours

    def draw_debug(self, contour, cx, cy, shape_class):
        cv2.drawContours(self.debug_img, [contour], 0, (0, 0, 255), 3)
        cv2.putText(
            self.debug_img,
            shape_class,
            (cx - 35, cy + 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    def unit_vector(self, v):
        return v / np.linalg.norm(v)

    def get_corner_angle(self, p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p1[0] - p3[0], p1[1] - p3[1]])
        v1_unit = self.unit_vector(v1)
        v2_unit = self.unit_vector(v2)
        radians = np.arccos(np.clip(np.dot(v1_unit, v2_unit), -1, 1))
        return math.degrees(radians)

    def find_shapes(self, epsilon_factor=0.02):
        self.get_binary_image()
        self.get_contours()
        for n, cnt in enumerate(self.contours):
            approx = cv2.approxPolyDP(
                cnt, epsilon_factor * cv2.arcLength(cnt, True), True
            )
            M = cv2.moments(approx)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            p1 = approx[0][0]
            p2 = approx[1][0]
            p3 = approx[-1][0]
            if len(approx) == 3:  # its a triangle!
                self.draw_debug(cnt, cx, cy, "Triangle")
            if len(approx) == 4:
                degrees = self.get_corner_angle(p1, p2, p3)
                print(degrees)
                if degrees == 90:
                    self.draw_debug(cnt, cx, cy, "Rectangle")
                else:
                    self.draw_debug(cnt, cx, cy, "Trapezoid")
            if len(approx) == 7:
                self.draw_debug(cnt, cx, cy, "Heptagon")
            if len(approx) == 10:
                self.draw_debug(cnt, cx, cy, "Star")
            if len(approx) == 12:
                self.draw_debug(cnt, cx, cy, "Cross")


if __name__ == "__main__":
    img = cv2.imread("shapes.jpg")
    shape_recognition = ShapeRecognition(img)

    shape_recognition.find_shapes()

    combined = np.vstack(
        (
            shape_recognition.debug_img,
            cv2.cvtColor(shape_recognition.binary_img, cv2.COLOR_GRAY2BGR),
        )
    )
    cv2.imshow("combined", combined)
    cv2.waitKey(0)
