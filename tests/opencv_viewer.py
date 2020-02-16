from RealSenseCamera import *
import numpy as np
import cv2

# Configure D415, enables depth + color streams
width, height = 640, 480
fps = 30
camera = RealSenseCamera(width, height, fps)

try:
	while True:
		depth_image, color_image = camera.get_images(asarray=True)
		color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

		images = np.hstack((color_image, depth_colormap))
		cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
		cv2.imshow("RealSense", images)
		cv2.waitKey(1)

finally:
	pipeline.stop()
