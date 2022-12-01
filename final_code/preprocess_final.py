import cv2 
import numpy as np
import argparse
import random as rng
import imutils
import os
from utils import *


def get_rectangle_coords(tup_coords):
	# top_left_x, top_left_y,bottom_right_x,bottom_right_y = tup_coords
	# width,height = bottom_right_x - top_left_x, top_left_y - bottom_right_y
	top_left_x, top_left_y,width,height = tup_coords
	bounding_rect = np.array([[[top_left_x,top_left_y]],[[top_left_x,top_left_y+height]],[[top_left_x+width,top_left_y+height]],[[top_left_x+width,top_left_y]]])
	return bounding_rect


code_path = code_path
image_path = os.path.join(code_path,"concrete_crack_images")

def crop_image():
	#for img in sorted([x for x in os.listdir(image_path) if x in ["00000232.jpg", "00000223.jpg", "00000415.jpg", "00000443.jpg", \
	#	"00000474.jpg", "00000262.jpg", "00000282.jpg", "00000324.jpg"]]):
	img_loc = os.path.join(image_path, img)
	image = cv2.imread(img_loc)
	# print(image.shape)

	ratio = image.shape[0] / 300.0
	orig = image.copy()
	image = imutils.resize(image, height = 300)
	# convert the image to grayscale, blur it, and find edges
	# in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 11, 17, 17)
	edged = cv2.Canny(gray, 30, 200)

	cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
	screenCnt = None

	# loop over our contours
	contour_map = {}
	count = 0

	for c in cnts:
		# approximate the contour
		# peri = cv2.arcLength(c, True)
		# approx = cv2.approxPolyDP(c, 0.015* peri, True)
		# # if our approximated contour has four points, then
		# # we can assume that we have found our screen

		# if len(approx) == 4:
		# 	screenCnt = approx
		# 	#print(screenCnt)
		# # 	break
		#else:


		rect_box = cv2.boundingRect(c)
		#cv2.rectangle(image, rect_box, (0, 0, 255))
		#get_rectangle_coords(rect_box)
		coords = get_rectangle_coords(rect_box)
		contour_map[rect_box] = [coords,cv2.contourArea(c)]
		# print(rect_box)
		# print(contour_map.values())

		if screenCnt is None:
			largest_area = max([x[1] for x in contour_map.values()])
			for key in contour_map.keys():
				if contour_map[key][1] == largest_area:
					screenCnt = contour_map[key][0]
					#print(screenCnt)
					cv2.rectangle(image, key, (0, 255, 255))
					break

			#print(get_rectangle_coords)
		# 	contour_map[count] = [approx,peri]
		# 	count +=1

	# second_l_peri = sorted([x[1] for x in contour_map.values()])[-2]
	# contour_key = [key for key in contour_map.keys() if contour_map[key][1] == second_l_peri]

	# screenCnt = contour_map[contour_key[0]][0]

	# print(screenCnt)

	# and bottom-left order

	pts = screenCnt.reshape(4, 2)
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point has the smallest sum whereas the
	# bottom-right has the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# compute the difference between the points -- the top-right
	# will have the minumum difference and the bottom-left will
	# have the maximum difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# multiply the rectangle by the original ratio
	rect *= ratio



	# now that we have our rectangle of points, let's compute
	# the width of our new image
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	# ...and now for the height of our new image
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	# take the maximum of the width and height values to reach
	# our final dimensions
	maxWidth = max(int(widthA), int(widthB))
	maxHeight = max(int(heightA), int(heightB))
	# construct our destination points which will be used to
	# map the screen to a top-down, "birds eye" view
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# calculate the perspective transform matrix and warp
	# the perspective to grab the screen
	M = cv2.getPerspectiveTransform(rect, dst)
	warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))



	# values of 0 and 255, respectively
	warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
	#warp = exposure.rescale_intensity(warp, out_range = (0, 255))
	# the pokemon we want to identify will be in the top-right
	# corner of the warped image -- let's crop this region out
	(h, w) = warp.shape
	(dX, dY) = (int(w * 0.9), int(h * 0.9))
	
	#temp value is 1, it used to be 10
	crop = warp[1:dY, w - dX:w - 1]
	return crop
	# save the cropped image to file
	# cv2.imwrite("output/cropped_{}.png".format(img.split(".")[0]), crop)
	
	# show our images
	#cv2.imshow("image", image)
	#cv2.imshow("edge", edged)
	#cv2.imshow("warp", imutils.resize(warp, height = 300))
	#cv2.imshow("crop", imutils.resize(crop, height = 300))
	#cv2.waitKey(0)

	#cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3) 
	#cv2.imshow("Game Boy Screen", image) 
	
for img in sorted(os.listdir(image_path)):

	#for img in sorted([x for x in os.listdir(image_path) if x in ["00000232.jpg", "00000223.jpg", "00000415.jpg", "00000443.jpg", \
	#	"00000474.jpg", "00000262.jpg", "00000282.jpg", "00000324.jpg"]]):
	img_loc = os.path.join(image_path, img)
	image = cv2.imread(img_loc)
	# print(image.shape)

	ratio = image.shape[0] / 300.0
	orig = image.copy()
	image = imutils.resize(image, height = 300)
	# convert the image to grayscale, blur it, and find edges
	# in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 11, 17, 17)
	edged = cv2.Canny(gray, 30, 200)

	cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
	screenCnt = None

	# loop over our contours
	contour_map = {}
	count = 0

	for c in cnts:
		# approximate the contour
		# peri = cv2.arcLength(c, True)
		# approx = cv2.approxPolyDP(c, 0.015* peri, True)
		# # if our approximated contour has four points, then
		# # we can assume that we have found our screen

		# if len(approx) == 4:
		# 	screenCnt = approx
		# 	#print(screenCnt)
		# # 	break
		#else:


		rect_box = cv2.boundingRect(c)
		#cv2.rectangle(image, rect_box, (0, 0, 255))
		#get_rectangle_coords(rect_box)
		coords = get_rectangle_coords(rect_box)
		contour_map[rect_box] = [coords,cv2.contourArea(c)]
		# print(rect_box)
		# print(contour_map.values())

		if screenCnt is None:
			largest_area = max([x[1] for x in contour_map.values()])
			for key in contour_map.keys():
				if contour_map[key][1] == largest_area:
					screenCnt = contour_map[key][0]
					#print(screenCnt)
					cv2.rectangle(image, key, (0, 255, 255))
					break

			#print(get_rectangle_coords)
		# 	contour_map[count] = [approx,peri]
		# 	count +=1

	# second_l_peri = sorted([x[1] for x in contour_map.values()])[-2]
	# contour_key = [key for key in contour_map.keys() if contour_map[key][1] == second_l_peri]

	# screenCnt = contour_map[contour_key[0]][0]

	# print(screenCnt)

	# and bottom-left order

	pts = screenCnt.reshape(4, 2)
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point has the smallest sum whereas the
	# bottom-right has the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# compute the difference between the points -- the top-right
	# will have the minumum difference and the bottom-left will
	# have the maximum difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# multiply the rectangle by the original ratio
	rect *= ratio



	# now that we have our rectangle of points, let's compute
	# the width of our new image
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	# ...and now for the height of our new image
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	# take the maximum of the width and height values to reach
	# our final dimensions
	maxWidth = max(int(widthA), int(widthB))
	maxHeight = max(int(heightA), int(heightB))
	# construct our destination points which will be used to
	# map the screen to a top-down, "birds eye" view
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# calculate the perspective transform matrix and warp
	# the perspective to grab the screen
	M = cv2.getPerspectiveTransform(rect, dst)
	warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))



	# values of 0 and 255, respectively
	warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
	#warp = exposure.rescale_intensity(warp, out_range = (0, 255))
	# the pokemon we want to identify will be in the top-right
	# corner of the warped image -- let's crop this region out
	(h, w) = warp.shape
	(dX, dY) = (int(w * 0.9), int(h * 0.9))
	
	#temp value is 1, it used to be 10
	crop = warp[1:dY, w - dX:w - 1]
	disp_img(crop)
	# save the cropped image to file
	# cv2.imwrite("output/cropped_{}.png".format(img.split(".")[0]), crop)
	
	# show our images
	#cv2.imshow("image", image)
	#cv2.imshow("edge", edged)
	#cv2.imshow("warp", imutils.resize(warp, height = 300))
	#cv2.imshow("crop", imutils.resize(crop, height = 300))
	#cv2.waitKey(0)

	#cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3) 
	#cv2.imshow("Game Boy Screen", image) 
	#cv2.waitKey(0)
