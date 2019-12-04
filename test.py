import cv2
import numpy as np
import math


def returnAngle(p1, p2, p3):

    vector1 = [(p1[0][0] - p2[0][0]), (p1[0][1] - p2[0][1])]
    vector2 = [(p1[0][0] - p3[0][0]), (p1[0][1] - p3[0][1])]

    cos = [((vector1[0] * vector2[0]) + (vector1[1] * vector2[1]))/((math.sqrt(math.pow(vector1[0], 2)
                                                                               )+math.pow(vector1[1], 2)) + (math.sqrt(math.pow(vector2[0], 2))+math.pow(vector2[1], 2)))]
    degree = math.degrees(cos[0])
    degree = 90 - degree

    return degree


# img = cv2.imread('smallb3.jpg') # quarter circle
# img = cv2.imread('smallo1.jpg') # Star
# img = cv2.imread('smallr3.jpg') # Triangle
img = cv2.imread('shapes.jpg')  # Shapes


frame = img.copy()

lower = np.array([0, 0, 0], dtype=np.uint8)
upper = np.array([15, 15, 15], dtype=np.uint8)
img = cv2.inRange(img, lower, upper)
kernel = np.ones((5, 5), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#img = cv2.erode(img, kernel, iterations=2)
img = cv2.dilate(img, kernel, iterations=1)

(flags, contours, h) = cv2.findContours(
    img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for n, cnt in enumerate(contours):
    # Number of curves in the contour
    approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
    M = cv2.moments(cnt)
    #minMaxCos = returnMinMaxAngle(approx)
    if M['m00'] > 300:
        if len(approx) >= 3 and len(approx) <= 3:  # for a triangle
            cx = int(M['m10']/M['m00'])  # x-position of center
            cy = int(M['m01']/M['m00'])  # y-position of center
            cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)
            cv2.putText(frame, "Triangle", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1, cv2.LINE_AA)
            # break
        if len(approx) >= 4 and len(approx) <= 4:  # for a square or rectangle
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            p1 = approx[0]
            p2 = approx[1]
            x = len(approx)
            p3 = approx[x-1]
            angle = abs(returnAngle(p1, p2, p3))
            if angle > 80 and angle < 100:
                cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)
                cv2.putText(frame, "Square", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)
                cv2.putText(frame, "Trapezoid", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1, cv2.LINE_AA)
            # break
        if len(approx) >= 5 and len(approx) <= 5:  # for a pentagon
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)
            cv2.putText(frame, "Pentagon", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1, cv2.LINE_AA)

            # break
        if len(approx) >= 6 and len(approx) <= 6:  # for a hexagon
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)
            cv2.putText(frame, "Hexagon", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1, cv2.LINE_AA)

            # break

        if len(approx) >= 7 and len(approx) <= 7:  # for a heptagon
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)
            cv2.putText(frame, "Heptagon", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1, cv2.LINE_AA)
            # break

        ''' 
        if len(approx)>=8 and len(approx)<=8: # for a octagon
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)
            cv2.putText(frame, "Octagon", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.25,(255,0,0),1,cv2.LINE_AA)      
        '''  # break
        if len(approx) >= 8 and len(approx) <= 12:  # from decagon to cross
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            p1 = approx[0]
            p2 = approx[1]
            x = len(approx)
            p3 = approx[x-1]
            angle = abs(returnAngle(p1, p2, p3))
            if angle > 65 and angle < 75:
                cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)
                cv2.putText(frame, "Star", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.25, (255, 0, 0), 1, cv2.LINE_AA)
            elif angle > 80 and angle < 100:
                cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)
                cv2.putText(frame, "Cross", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.25, (255, 0, 0), 1, cv2.LINE_AA)
            elif angle > 140 and angle < 150:
                cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)
                cv2.putText(frame, "Decagon", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1, cv2.LINE_AA)
            elif angle > 168 and angle < 173:
                cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)
                cv2.putText(frame, "Octogon", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1, cv2.LINE_AA)
            area = cv2.contourArea(cnt)

            px = p1[0][0]
            py = p2[0][1]
            distance = math.sqrt(math.pow((px + cx), 2) +
                                 math.pow((py + cy), 2))

            calcArea = 2*3.14*(math.pow(distance, 2))
            if (calcArea/area)*100 < 10:
                cv2.putText(frame, "Circle", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1, cv2.LINE_AA)
            # break

        if len(approx) > 12:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)
            cv2.putText(frame, "Circle", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1, cv2.LINE_AA)


cv2.imshow('Image 1', img)
cv2.imshow('Image 2 contours', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
