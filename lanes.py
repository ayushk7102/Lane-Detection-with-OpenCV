import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def canny(f):
	gray = cv.cvtColor(f, cv.COLOR_BGR2GRAY)

	blur = cv.GaussianBlur(gray, (5,5), 0)

	canny = cv.Canny(blur, 50, 150)
	#cv.imshow('SA', f)

	#cv.imshow('Final', canny)

def roi(f):
	ht = f.shape[0]
	polys = np.array([[(130, ht-40), (1100, ht-40), (704, 422)]])
	polys = np.array([[(130, ht-40), (1100, ht-40), (743, 433), (652, 420)]])
	mask = np.zeros_like(f)
	cv.fillPoly(mask, polys, 255)
	masked_img = cv.bitwise_and(f, mask) 

	return masked_img
	
def display_lines(f, lines):
	line_img = np.zeros_like(f)
	if lines is not None:
		for x1, y1, x2, y2 in lines:
			
			cv.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
		return line_img

def make_coords(img, line_params):
	global prev_line_params
	if(np.isnan(np.sum(line_params))):
		line_params = prev_line_params
	print(line_params)
	if line_params.size == 2: 
		m, b = line_params

		prev_line_params = np.copy(line_params)
	else:
		m, b = prev_line_params

	y1 = img.shape[0]
	y2 = int(y1*(3/5))

	x1 = int((y1 - b)/m)
	x2 = int((y2 - b)/m)

	return np.array([x1, y1, x2, y2])

def average_lines(pre_lanes, lines):
	left_fit = []
	right_fit = []

	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		params = np.polyfit((x1, x2), (y1, y2), deg=1)
		#print(params)
		slope = params[0]
		intercept = params[1]

		if slope < 0:
			left_fit.append((slope, intercept))
		else : 
			right_fit.append((slope, intercept))

	left_fit_avg = np.average(left_fit, axis=0)

	right_fit_avg = np.average(right_fit, axis=0)



	left_line = make_coords(pre_lanes, left_fit_avg)
	right_line = make_coords(pre_lanes, right_fit_avg)

	return np.array([left_line, right_line])

#cap = cv.VideoCapture('/home/ayush/Desktop/Manas/OpenCV/Lane Detection/challenge_video.mp4')


cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(8)

cap = cv.VideoCapture('/home/ayush/Desktop/Manas/OpenCV/Lane Detection/challenge_video.mp4')
frameCount  = 0
while cap.isOpened():
	ret, f = cap.read()
	#f = cv.imread('/home/ayush/Desktop/Manas/OpenCV/Lane Detection/sample frame.png')

	
	pre_lanes = np.copy(f)

	gray = cv.cvtColor(f, cv.COLOR_BGR2GRAY)

	blur = cv.GaussianBlur(gray, (5,5), 0)

	canny = cv.Canny(blur, 50, 150)

	cropd = roi(canny)
	#plt.imshow(cropd)
	#plt.show()
	lines = cv.HoughLinesP(cropd, 2, np.pi/180, 100, np.array([]), minLineLength=50, maxLineGap=5)

	avg_lines = average_lines(pre_lanes, lines)


#	line_img = display_lines(f, lines)
	line_img = display_lines(f, avg_lines)

	#cv.imshow("winname", line_img)
	combined = cv.addWeighted(pre_lanes, 0.8, line_img, 1, 1)
	frameCount+=1

	print(frameCount)
	

	cv.imshow("result", combined)
	if cv.waitKey(5) & 0xFF == ord('q'):
	 	break

cap.release()
cv.destroyAllWindows()

