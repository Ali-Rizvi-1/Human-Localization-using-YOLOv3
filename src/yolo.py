#!/usr/bin/env python
from __future__ import print_function
from glob import glob
from modulefinder import packagePathMap

from numpy import imag

import roslib
import sys
import rospy
import cv2
# from cv2 import image
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
# from geometry_msgs.msg import mapData
from sensor_msgs.msg import mapData
# from std_msgs.msg import mapData
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates

import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import PIL

from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.patches import Rectangle
import time
import math
import Boundbox

from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

print('Imports completed')

#define parameters for YOLO

# define the anchors
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

# define the probability threshold for detected objects
class_threshold = 0.6

# define the labels
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
	"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

print('About to start loading the model')
s = time.time() 
model = keras.models.load_model('/home/ali/catkin_ws/src/yolov3.h5') #'/home/ali/catkin_ws/src/human-detection-yolo/src
print('Loaded pretrained model in', str(time.time() - s ),'s')

bridge_rgb = CvBridge()
bridge_depth = CvBridge()
global_img = Image()
global_depth = Image()
# cv_image_rgb = None
rgb_image_flag = 0
depth_image_flag = 0
print('objects created')
q0, q1, q2, q3, x1, y1, z1 = 0, 0, 0, 0, 0, 0, 0

def image_callback(msg):
  '''
  Save the rgb image in global variable.
  '''
  global global_img
  global rgb_image_flag
  global_img = msg
  rgb_image_flag = 1
  
def depth_callback(msg):
  '''
  Save the depth image in global variable.
  '''
  global global_depth 
  global depth_image_flag 
  global_depth = msg
  depth_image_flag = 1

def slam_callback(data):
  global q0, q1, q2, q3, x1, y1, z1
  q0 = data.pose.pose.orientation.w
  q1 = data.pose.pose.orientation.x
  q2 = data.pose.pose.orientation.y
  q3 = data.pose.pose.orientation.z
  x1 = data.pose.pose.position.y*-1
  y1 = data.pose.pose.position.x
  z1 = data.pose.pose.position.z

def slam_mapData_callback(data):
	print("Map data received from rtabmap")
	pass

def gazebo_callback(data):
	global q0, q1, q2, q3, x1, y1, z1
	# print('gazebo callback')  
	for i in range(len(data.name)):
		if data.name[i] == 'iris':
			ind = i
			break
	#print(ind)
	# print(data.pose[ind])
	q0 = data.pose[ind].orientation.w
	q1 = data.pose[ind].orientation.x
	q2 = data.pose[ind].orientation.y
	q3 = data.pose[ind].orientation.z
	x1 = data.pose[ind].position.x
	y1 = data.pose[ind].position.y
	z1 = data.pose[ind].position.z

# functions used after calling model 
def _sigmoid(x):
  return 1. /(1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):

	grid_h, grid_w = netout.shape[:2]
	nb_box = 3
	netout = netout.reshape((grid_h, grid_w, nb_box, -1))
	nb_class = netout.shape[-1] - 5
	boxes = []
	netout[..., :2]  = _sigmoid(netout[..., :2])
	netout[..., 4:]  = _sigmoid(netout[..., 4:])
	netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
	netout[..., 5:] *= netout[..., 5:] > obj_thresh
 
	for i in range(grid_h*grid_w):
		row = i / grid_w
		col = i % grid_w
		for b in range(nb_box):
			# 4th element is objectness score
			objectness = netout[int(row)][int(col)][b][4]
			if(objectness.all() <= obj_thresh): continue
			# first 4 elements are x, y, w, and h
			x, y, w, h = netout[int(row)][int(col)][b][:4]
			x = (col + x) / grid_w # center position, unit: image width
			y = (row + y) / grid_h # center position, unit: image height
			w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
			h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
			# last elements are class probabilities
			classes = netout[int(row)][col][b][5:]
			box = Boundbox.BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
			boxes.append(box)
	return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
	new_w, new_h = net_w, net_h
	for i in range(len(boxes)):
		x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
		y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
		boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
		boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
		boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
		boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
	x1, x2 = interval_a
	x3, x4 = interval_b
	if x3 < x1:
		if x4 < x1:
			return 0
		else:
			return min(x2,x4) - x1
	else:
		if x2 < x3:
			return 0
		else:
			return min(x2,x4) - x3
 
def bbox_iou(box1, box2):
	intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
	intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
	intersect = intersect_w * intersect_h
	w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
	w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
	union = w1*h1 + w2*h2 - intersect
	return float(intersect) / union
 
def do_nms(boxes, nms_thresh):
	if len(boxes) > 0:
		nb_class = len(boxes[0].classes)
	else:
		return
	for c in range(nb_class):
		sorted_indices = np.argsort([-box.classes[c] for box in boxes])
		for i in range(len(sorted_indices)):
			index_i = sorted_indices[i]
			if boxes[index_i].classes[c] == 0: continue
			for j in range(i+1, len(sorted_indices)):
				index_j = sorted_indices[j]
				if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
					boxes[index_j].classes[c] = 0

# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
	v_boxes, v_labels, v_scores = list(), list(), list()
	# enumerate all boxes
	for box in boxes:
		# enumerate all possible labels
		for i in range(len(labels)):
			# check if the threshold for this label is high enough
			if box.classes[i] > thresh:
				v_boxes.append(box)
				v_labels.append(labels[i])
				v_scores.append(box.classes[i]*100)
				# don't break, many labels may trigger for one box
	return v_boxes, v_labels, v_scores

def load_image(rgb_image, desired_shape):
	global bridge_rgb
	# global cv_image_rgb
	print(type(rgb_image))
	# print(len(rgb_image.data))
	# print(rgb_image)
	try:
		print('RGB image converted to cv image')
		cv_image_rgb = bridge_rgb.imgmsg_to_cv2(rgb_image, "bgr8")#desired_encoding="bgr8"
	except CvBridgeError as e:
		print(e)
	print(type(cv_image_rgb))
	(height,width,channels) = cv_image_rgb.shape
	cv_image_rgb = cv2.resize(cv_image_rgb, desired_shape, interpolation = cv2.INTER_AREA )
	cv_image_rgb = img_to_array(cv_image_rgb)
	cv_image_rgb = cv_image_rgb.astype('float32')
	cv_image_rgb /= 255.0
	cv_image_rgb = expand_dims(cv_image_rgb,0) 
	return cv_image_rgb, width, height

image_centroid = Float64MultiArray()
global_centroid = Float64MultiArray()
global_centroid_list = []
local_centroid = Float64MultiArray()
local_centroid_list = []

def use_model(image,input_h,input_w,image_h,image_w):
	start= time.time()
	yhat = model.predict(image)
	end = time.time()
	print("Total time for Model to run: {:.1f} s".format(end-start))

	# summarize the shape of the list of arrays
	# print([a.shape for a in yhat])

	start= time.time()
	boxes = list() 
	for i in range(len(yhat)):
		# decode the output of the network
		boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)

	end = time.time()
	print("Total time for decoding boxes boxes: {:.1f} s".format(end-start))

	start =time.time()
	# correct the sizes of the bounding boxes for the shape of the image
	correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
	end =time.time()
	print("Total time to correct bounding boxes: {:.1f} s".format(end-start))
																
	start =time.time()
	# suppress non-maximal boxes
	do_nms(boxes, 0.5)
	end =time.time()
	print("Total time to suppress non-maximal boxes: {:.1f} s".format(end-start))

	centroid_temp = []
	v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

	global image_centroid 
	print('v_boxes',len(v_boxes),v_boxes)
	for i in range(len(v_boxes)):
		if v_labels[i] == 'person':
			# get coordinates
			y1, x1, y2, x2 = v_boxes[i].ymin, v_boxes[i].xmin, v_boxes[i].ymax, v_boxes[i].xmax
			# calculate width and height of the box
			width, height = x2 - x1, y2 - y1
			# drawing boounding boxes
			# cv2.rectangle(cv_image, (x1,y1), (x1+width,y1+height), (0,255,0), 2) 
			# print(v_boxes[i].xmin, v_boxes[i].xmax, v_boxes[i].get_label() )
			centeroid_x = (x1+x2)/2
			centeroid_y = (y1+y2)/2
			centroid_temp.append(centeroid_x)
			centroid_temp.append(centeroid_y)
	image_centroid.data = centroid_temp

	# summarize what we found
	for i in range(len(v_boxes)):
		print(v_labels[i], v_scores[i])

	# draw what we found
	# draw_boxes(image, v_boxes, v_labels, v_scores)

	# cv2.imwrite('/home/ali/catkin_ws/src/human-detection-yolo/src/result2.png',cv_image)
	# cv2.imwrite('/home/ali/catkin_ws/src/human-detection-yolo/src/result2.jpeg',image[0])
	# cv2.imshow("Image window",image)
	# cv2.waitKey(3)
	return image_centroid

def local_transformation(depth_img):
	global bridge_depth
	try:
		print('Depth image converted to cv image')
		depth_image = bridge_depth.imgmsg_to_cv2(depth_img, "passthrough")
	except CvBridgeError as e:
		print(e)

	global local_centroid
	global local_centroid_list

	image_centroid_data = image_centroid.data
	print('Image centroid list in local transformation funct',image_centroid_data)
	for k in range(len(image_centroid_data)/2):
		
		if (k%2.0 )== 0.0:
			a = 10000000.0
			b = 10000000.0
			r = int(image_centroid_data[k+1])
			c = int(image_centroid_data[k])

			(rows,cols) = depth_image.shape
			#create numpy list of image for simplicity
			lst = []
			for i in range(0,rows-1):
				tlst = []
				for j in range(0,cols-1):
					tlst.append(depth_image[i,j])
				lst.append(tlst)

			lst = np.array(lst)
			# Gray image is only for visualization, for all purposes,
			# use lst to get depth values from image
			uint_img = np.array(lst/255).astype('uint8')
			gray = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)

			z = float(lst[r,c]) #+620.0#/ 1000.0 # in meters  
			# print('Depth of the centroid')
			
			# localization local coor
			# f is focal length and ppi is pixels per inch
			# l_p = pixes/ppi
			# l_p = lp*    # covert to meters
			cx = 320.0
			fx = 343.496
			ry = 320.0
			fy = 343.496
			u=float(c)
			x = z*(c - cx)/fx 
			y = z*(r - ry)/fy 
			# global coordinates
			#(x,y)

			# visualize - marker array
			# red big dabba at (x,y)
			a = z/1000.0
			b = x/1000.0
			c = y/1000.0
			#print(x,y)
			
			print("x = ", z/1000.0, "and y = ", x/1000.0, "and z = ", y/1000.0)

			if (local_centroid_list) != 0:
				flag_x, flag_y = 0, 0
				for h in range(len(local_centroid_list)):
					if (h%2) == 0.0:
						if local_centroid_list[h]-2 < a < local_centroid_list[h]+2: 
							flag_x = 1
						if local_centroid_list[h+1]-2 < b < local_centroid_list[h+1]+2: 
							flag_y = 1	
				if flag_x == 0 or flag_y == 0:
					if a != 10000000.0 and b != 10000000.0: # and c != 0.0
						local_centroid_list.append(a)
						local_centroid_list.append(b)
			else:
				if a != 10000000.0 and b != 10000000.0: # and c != 0.0
					local_centroid_list.append(a)
					local_centroid_list.append(b)
        
	local_centroid.data = local_centroid_list
	print('local coordinates',local_centroid)
	return local_centroid

def global_transformation():
  print('slam data',q0, q1, q2, q3, x1, y1, z1)
  global global_centroid
  global global_centroid_list
  yaw   = np.arctan2(2.0 * (q3 * q0 + q1 * q2) , - 1.0 + 2.0 * (q0 * q0 + q1 * q1))
  print('Yaw', yaw)
  local_centroid_data = local_centroid.data
  print('Local centroid list in global transformation funct',local_centroid_data)
  if len(local_centroid_data) != 0: # check to see if there are any humans detected
    for j in range(len(local_centroid_data)):
      if (j%2.0) == 0.0:
        Tx = local_centroid_data[j]*math.cos(yaw) + local_centroid_data[j+1]*math.sin(yaw) + x1
        Ty = local_centroid_data[j]*math.sin(yaw) - local_centroid_data[j+1]*math.cos(yaw) + y1
        
        if len(global_centroid_list) != 0:
          flag_x = 0
          flag_y = 0
          for i in range(len(global_centroid_list)):
            if (i%2.0) == 0.0:
              if Tx-2.5 < global_centroid_list[i] < Tx+2.5:
                flag_x = 1
              if Ty-2.5 < global_centroid_list[i+1] < Ty+2.5 :
                flag_y = 1
          if flag_x == 0 or flag_y == 0:
            global_centroid_list.append(Tx)
            global_centroid_list.append(Ty)

          #print([flag_x, flag_y])
        else:
          global_centroid_list.append(Tx) #[Tx,Ty]
          global_centroid_list.append(Ty)

  global_centroid.data = global_centroid_list
  print('global coordinates',global_centroid)
  return global_centroid

def global_transformation_gazebo():

	print('gazebo data',q0, q1, q2, q3, x1, y1, z1)
	global global_centroid
	global global_centroid_list
	yaw   = np.arctan2(2.0 * (q3 * q0 + q1 * q2) , - 1.0 + 2.0 * (q0 * q0 + q1 * q1))
	local_centroid_data = local_centroid.data
	if len(local_centroid_data) != 0:
		for j in range(len(local_centroid_data)):
			if (j%2.0) == 0.0:
				Tx = local_centroid_data[j]*math.cos(yaw) + local_centroid_data[j+1]*math.sin(yaw) + x1
				Ty = local_centroid_data[j]*math.sin(yaw) - local_centroid_data[j+1]*math.cos(yaw) + y1
			
				if len(global_centroid_list) != 0:
					flag_x = 0
					flag_y = 0
					for i in range(len(global_centroid_list)):
						if (i%2.0) == 0.0:
							if Tx-2.0 < global_centroid_list[i] < Tx+2.0:
								flag_x = 1
							if Ty-2.0 < global_centroid_list[i+1] < Ty+2.0 :
								flag_y = 1
					if flag_x == 0 or flag_y == 0:
						global_centroid_list.append(Tx)
						global_centroid_list.append(Ty)
				else:
					global_centroid_list.append(Tx) #[Tx,Ty]
					global_centroid_list.append(Ty)

	global_centroid.data = global_centroid_list
	print('global gazebo coordinates',global_centroid)
	return global_centroid

image_sub = rospy.Subscriber('/camera/color/image_raw',Image, image_callback)
depth_sub = rospy.Subscriber('/camera/depth_aligned_to_color_and_infra1/image_raw',Image, depth_callback)
slam_pose_sub = rospy.Subscriber("/rtabmap/odom", Odometry, slam_callback)
slam_mapData_sub = rospy.Subscriber("/rtabmap/mapData", mapData, slam_mapData_callback)
# drone_pose_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, gazebo_callback)

# centroid_pub = rospy.Publisher('/centroid', Float64MultiArray)
image_centroid_pub = rospy.Publisher('/image_centroid', Float64MultiArray, queue_size=1)
local_centroid_pub = rospy.Publisher('/local_centroid', Float64MultiArray, queue_size=1)
global_centroid_pub = rospy.Publisher('human_detection/human_locations', Float64MultiArray, queue_size=1)#/global_centroid

# start a node 
rospy.init_node('HumanDetection')

# next line simply sets a rate for you node.
rate = rospy.Rate(10)
''' 
'''

while not rospy.is_shutdown():
	
	if rgb_image_flag == 1 and depth_image_flag == 1:
		#Take image and depth and save into diff variables at the same instant
		img = global_img
		depth = global_depth

		#Run model, deect boxes, and Find centroid
		#   image, image_w, image_h = load_image_pixels(img, (416, 416))
		detected_image, image_w, image_h = load_image(img, (416, 416))
		input_w = 416
		input_h = 416
		image_centroid = use_model(detected_image,input_h,input_w,image_h,image_w)
		
		# local transformation 
		local_centroid = local_transformation(depth)

		# global transformation
		global_centroid = global_transformation()
		# global_centroid = global_transformation_gazebo()

		# publishing 
		image_centroid_pub.publish(image_centroid)
		local_centroid_pub.publish(local_centroid)
		global_centroid_pub.publish(global_centroid)
	
	rate.sleep()
