import cv2
import numpy as np
import math

class ShapeRecognition()
	
	def __init__(self, img):
		self.img = img
		self.contours = None

	def preProcessing(self):
        lower = np.array([0,0,0], dtype=np.uint8) # Lower range of black
        upper = np.array([15,15,15], dtype=np.uint8) # Upper range of black
        mask = cv2.inRange(self.shapeImg, lower, upper) # Only display pixels within the lower and upper range
		cv2.imshow("mask", mask)

		(flags, self.contours, h) = cv2.findContours(self.shapeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		return self.contours

	def returnAngle(self, p1, p2, p3):
	    
	    vector1 = [(p1[0][0] - p2[0][0]), (p1[0][1] - p2[0][1])]
	    vector2 = [(p1[0][0] - p3[0][0]), (p1[0][1] - p3[0][1])]
	    
	    cos = [((vector1[0] * vector2[0]) + (vector1[1] * vector2[1]))/((math.sqrt(math.pow(vector1[0],2))+math.pow(vector1[1],2)) + (math.sqrt(math.pow(vector2[0],2))+math.pow(vector2[1],2)))]
	    degree = math.degrees(cos[0])
	    degree = 90 - degree
	    
	    return degree

	def findShapes(self):
		for n, cnt in enumerate(self.contours):
		    approx = cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt,True),True)#Number of curves in the contour
		    M = cv2.moments(cnt)
		    #minMaxCos = returnMinMaxAngle(approx)
		    if M['m00'] > 300:
		        if len(approx)>=3 and len(approx)<=3: # for a triangle
		            cx = int(M['m10']/M['m00'])#x-position of center
		            cy = int(M['m01']/M['m00'])#y-position of center
		            cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)
		            cv2.putText(frame, "Triangle", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.25,(255,0,0),1,cv2.LINE_AA) 
		            #break
		        if len(approx)>=4 and len(approx)<=4: # for a square or rectangle
		            cx = int(M['m10']/M['m00'])
		            cy = int(M['m01']/M['m00'])
		            p1 = approx[0]
		            p2 = approx[1]
		            x = len(approx)
		            p3 = approx[x-1]
		            angle = abs(returnAngle(p1, p2, p3))
		            if angle > 80 and angle < 100:
		                cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)
		                cv2.putText(frame, "Square", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.25,(255,0,0),1,cv2.LINE_AA)
		                print "Square of angle: ", angle
		            else:
		                cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)
		                cv2.putText(frame, "Trapezoid", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.25,(255,0,0),1,cv2.LINE_AA)
		                print "Trapezoid of angle: ", angle
		            #break
		         
		        if len(approx)>=7 and len(approx)<=7: # for a heptagon
		            cx = int(M['m10']/M['m00'])
		            cy = int(M['m01']/M['m00'])
		            cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)
		            cv2.putText(frame, "Heptagon", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.25,(255,0,0),1,cv2.LINE_AA)      
		            #break
		                          
		        if len(approx)>=10 and len(approx)<=12: # from decagon to cross
		            cx = int(M['m10']/M['m00'])
		            cy = int(M['m01']/M['m00'])
		            p1 = approx[0]
		            p2 = approx[1]
		            x = len(approx)
		            p3 = approx[x-1]
		            angle = abs(returnAngle(p1, p2, p3))
		            if angle > 65 and angle < 75:
		                cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)
		                cv2.putText(frame, "Star", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.25,(255,0,0),1,cv2.LINE_AA)
		                print "Star of angle:", angle
		            elif angle > 80 and angle < 100:
		                cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)
		                cv2.putText(frame, "Cross", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.25,(255,0,0),1,cv2.LINE_AA)                
		                print "Cross ", n, " has angle: ", angle  


img = cv2.imread("shapes.jpg")
shapeRecognition = ShapeRecognition(img)


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows