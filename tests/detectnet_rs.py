import jetson.inference
import jetson.utils

from RealSenseCamera import *
import numpy as np
import cv2

import argparse
import sys
import time

parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
					formatter_class=argparse.RawTextHelpFormatter,
					epilog=jetson.inference.detectNet.Usage())

parser.add_argument("--network", type=str, default="ssd-mobilenet-v2",
					help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf",
					help="detection overlay flags (e.g.--overlay=box,labels,conf)\nvalid\
					combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5,
					help="minimum detection threshold to use") 
parser.add_argument("--width", type=int, default=1280, help="desired width of\
					camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of\
					camera stream (default is 720 pixels)")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# Init RealSense camera (D415) and object detection network
camera = RealSenseCamera(opt.width, opt.height, fps=30)
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# display = jetson.utils.glDisplay()

while True:
# while display.IsOpen():
	# Sample current time for fps calc later
	start = time.time()	
	
	# Retrieve image frames from camera
	depth_img, color_img = camera.get_images() # add "asarray=True" to get img data as numpy array
	# Convert color_img to numpy array, keep depth as 'frame' object
	color_img_data = camera.get_image_data(color_img)
	img = cv2.cvtColor(color_img_data, cv2.COLOR_BGR2RGBA).astype(np.float32)
	# Copy color img data to CUDA mem
	img = jetson.utils.cudaFromNumpy(img) 
	jetson.utils.cudaDeviceSynchronize()

	input_width = color_img_data.shape[1]
	input_height = color_img_data.shape[0]
	# Retrieve object detection results
	detections = net.Detect(img, input_width, input_height, opt.overlay)
	overlay_img = jetson.utils.cudaToNumpy(img, input_width, input_height, 4)
	overlay_img = cv2.cvtColor(overlay_img.astype(np.uint8), cv2.COLOR_RGBA2RGB)
	#print("detected {:d} objects in image".format(len(detections)))
	cv2.namedWindow("hello", cv2.WINDOW_AUTOSIZE)
	font = cv2.FONT_HERSHEY_SIMPLEX

	for detection in detections:
		#print(net.GetClassDesc(detection.ClassID), detection.Confidence)
		center = detection.Center
		x, y = int(center[0]), int(center[1])
		# Pull z-distance from object's 2D center
		distance = round(depth_img.get_distance(x, y), 2)
		cv2.putText(overlay_img, "dist = "+str(distance), (x-15, y), 
					font, 0.8, (255,255,255), 1, cv2.LINE_AA)
	
	end = time.time()
	total_fps = round(1/(end-start), 3)
	nn_fps = round(net.GetNetworkFPS(), 3)
	total_txt = "Total FPS: {}".format(total_fps)
	title = "{:s} | Network FPS {:.0f}".format(opt.network, nn_fps)
	cv2.putText(overlay_img, total_txt, (15,15), font, 0.4, (255,255,255), 1, cv2.LINE_AA)
	cv2.putText(overlay_img, title, (15,30), font, 0.4, (255,255,255), 1, cv2.LINE_AA)
	#print(title)
	cv2.imshow("hello", overlay_img)
	cv2.waitKey(1)
	#display.RenderOnce(input_img, input_width, input_height)
	#display.SetTitle(title)
	#net.PrintProfilerTimes()
