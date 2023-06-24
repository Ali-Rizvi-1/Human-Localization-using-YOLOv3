#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, Point
from cv_bridge import CvBridge
from person_detector import PersonDetector
import cv2
from PIL import Image as PILImage
import numpy as np

class HumanDetectionNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('human_detection_node', anonymous=True)

        # Create a publisher for human centroids
        self.centroid_pub = rospy.Publisher('human_detection/human_centroids', PointStamped, queue_size=10)

        # Create a subscriber for color image
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        # Create a subscriber for depth image
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)

        # Create a subscriber for odometry
        self.odom_sub = rospy.Subscriber('/rtabmap/odom', Odometry, self.odom_callback)

        # Create an instance of PersonDetector
        self.detector = PersonDetector('weights/yolov5s.pt')

        # Create a bridge object for image conversion
        self.bridge = CvBridge()

        # Transformation matrices
        self.local_transform = np.array([[1.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 1.0]])  # Local transformation matrix
        self.global_transform = np.array([[1.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0],
                                          [0.0, 0.0, 1.0]])  # Global transformation matrix

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Convert OpenCV image to PIL image
            pil_image = PILImage.fromarray(cv_image)

            # Save the image temporarily (assuming you have write permissions)
            pil_image.save('/path/to/temp_image.jpg')

            # Detect people in the image
            bounding_boxes = self.detector.detect_person('/path/to/temp_image.jpg')

            # Calculate centroids for each bounding box
            centroids = []
            for bbox in bounding_boxes:
                centroid = self.calculate_centroid(bbox)
                local_centroid = self.transform_local_coordinates(centroid)
                global_centroid = self.transform_global_coordinates(local_centroid)
                centroids.append(global_centroid)

            # Publish the centroids as PointStamped messages
            self.publish_centroids(centroids)

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def depth_callback(self, msg):
        # Process depth image here to update local transformation matrix
        # You can extract depth values and perform calculations to update the local_transform matrix
        pass

    def odom_callback(self, msg):
        # Process odometry here to update global transformation matrix
        # You can extract position and orientation information from the odometry message
        # and perform calculations to update the global_transform matrix
        pass

    def calculate_centroid(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        centroid_x = int((x_max + x_min) / 2)
        centroid_y = int((y_max + y_min) / 2)
        return (centroid_x, centroid_y)

    def transform_local_coordinates(self, point):
        local_point = np.array([point[0], point[1], 1])
        transformed_local_point = np.dot(self.local_transform, local_point)
        return (transformed_local_point[0], transformed_local_point[1])

    def transform_global_coordinates(self, point):
        global_point = np.array([point[0], point[1], 1])
        transformed_global_point = np.dot(self.global_transform, global_point)
        return (transformed_global_point[0], transformed_global_point[1])

    def publish_centroids(self, centroids):
        for centroid in centroids:
            point_msg = PointStamped()
            point_msg.header.stamp = rospy.Time.now()
            point_msg.point = Point(x=centroid[0], y=centroid[1], z=0.0)
            self.centroid_pub.publish(point_msg)

    def run(self):
        # Start the main loop
        rospy.spin()

if __name__ == '__main__':
    try:
        node = HumanDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
