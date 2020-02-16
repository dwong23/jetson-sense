import pyrealsense2 as rs
import numpy as np

class RealSenseCamera():
	def __init__(self, width, height, fps):
		self.res_width = width
		self.res_height = height
		self.fps = fps
		self.pipeline = rs.pipeline()
		self.config = rs.config()
		self.enable_streams()
		self.align = rs.align(rs.stream.color)
		self.pipeline.start(self.config)

	def enable_streams(self, depth=True, color=True):
		if depth:
			self.config.enable_stream(rs.stream.depth, self.res_width,
										self.res_height, rs.format.z16, self.fps)
		if color:
			self.config.enable_stream(rs.stream.color, self.res_width,
										self.res_height, rs.format.rgb8, self.fps)
		
	def get_images(self, asarray=False):
		depth, color = None, None
		while not depth or not color:
			frames = self.pipeline.wait_for_frames()
			aligned = self.align.process(frames)
			#depth = frames.get_depth_frame()
			#color = frames.get_color_frame()
			depth = aligned.get_depth_frame()
			color = aligned.get_color_frame()

		if asarray:
			depth = self.get_image_data(depth)
			color = self.get_image_data(color)
		
		return depth, color

	def get_image_data(self, frame):
		return np.asanyarray(frame.get_data())


		
