#!/usr/bin/env python

from typing import Tuple
import tf
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, Point
from cv_bridge import CvBridge
from person_detector import PersonDetector
import cv2
from PIL import Image as PILImage
import numpy as np
import math

class HumanDetectionNode:
    def __init__(self):

        # Create a publisher for human centroids
        self.centroid_pub = rospy.Publisher('human_detection/human_locations', PointStamped, queue_size=10)

        # Create a subscriber for color image
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        # Create a subscriber for depth image
        self.depth_sub = rospy.Subscriber('/camera/depth_aligned_to_color_and_infra1/image_raw', Image, self.depth_callback)

        # Create a subscriber for odometry
        self.odom_sub = rospy.Subscriber('/rtabmap/odom', Odometry, self.odom_callback)

        # Create an instance of PersonDetector
        self.detector = PersonDetector('weights/yolov5s.pt')

        # Create a bridge object for image conversion
        self.bridge = CvBridge()

        # main logic of the node goes here! 

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Convert OpenCV image to PIL image
            pil_image = PILImage.fromarray(cv_image)

            # Detect people in the image
            bounding_boxes = self.detector.detect_person(pil_image)

            # Calculate centroids for each bounding box
            centroids = []
            for bbox in bounding_boxes:
                centroid_point = self.calculate_centroid(bbox)
                local_centroid_point = self.transform_local_coordinates(depth_img=self.depth_image,
                                                                        centroid_point=centroid_point)
                global_centroid = self.transform_global_coordinates(pose=self.pose,
                                                                    quaternion=self.quaternion,
                                                                    local_centroid_point=local_centroid_point)
                if global_centroid not in centroids:
                    centroids.append(global_centroid)

            # Publish the centroids as PointStamped messages
            self.publish_centroids(centroids)

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def depth_callback(self, msg):
        # Process depth image here to update local transformation matrix
        # You can extract depth values and perform calculations to update the local_transform matrix
        try:
            # Convert ROS Image message to OpenCV image
            cv_image_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")

            self.depth_image = cv_image_depth

        except CvBridgeError as e:
            print(e)

    def odom_callback(self, odom):
        """
        Callback function for odometry subscriber.
        """

        self.pose = (
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z
        )
        self.quaternion = (
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w
        )

    def calculate_centroid(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        centroid_x = int((x_max + x_min) / 2)
        centroid_y = int((y_max + y_min) / 2)
        return (centroid_x, centroid_y)
    
    def transform_local_coordinates(self, depth_img, centroid_point: Tuple[float, float]) -> Tuple[float, float]:
        try:
            print('Converting depth image to cv image')
            depth_image = self.bridge.imgmsg_to_cv2(depth_img, "passthrough")
        except CvBridgeError as e:
            raise ValueError('Error converting depth image to cv image') from e

        r, c = int(centroid_point[1]), int(centroid_point[0])

        try:
            z = depth_image[r, c]  # Depth value in millimeters
        except IndexError:
            raise ValueError('Invalid centroid point') from None

        # Localization in local coordinates
        cx, fx, ry, fy = 320.0, 343.496, 320.0, 343.496
        x = z * (c - cx) / fx
        y = z * (r - ry) / fy

        # Global coordinates (x, y)
        a, b = z / 1000.0, x / 1000.0  # Convert to meters

        print(f"x = {a} and y = {b} and z = {y / 1000.0}")

        # if not any(local_centroid_list[h] - 2 < a < local_centroid_list[h] + 2
        #         for h in range(0, len(local_centroid_list), 2)):

        if a != 10000000.0 and b != 10000000.0:
            local_centroid_point = (a, b)
            return local_centroid_point

        return None
    
    def transform_global_coordinates(self, pose: Tuple[float, float, float], quaternion: Tuple[float, float, float, float], local_centroid_point: Tuple[float, float]) -> Tuple[float, float]:
        try:
            x1, y1, z1 = pose[1] * -1, pose[0], pose[2]
            q0, q1, q2, q3 = quaternion[3], quaternion[0], quaternion[1], quaternion[2]

            yaw = np.arctan2(2.0 * (q3 * q0 + q1 * q2), -1.0 + 2.0 * (q0 * q0 + q1 * q1))
            print('Yaw:', yaw)

            if local_centroid_point:
                Tx = local_centroid_point[0] * math.cos(yaw) + local_centroid_point[1] * math.sin(yaw) + x1
                Ty = local_centroid_point[0] * math.sin(yaw) - local_centroid_point[1] * math.cos(yaw) + y1

                # flag_x, flag_y = False, False
                # for i in range(0, len(global_centroid_list), 2):
                #     if Tx - 2.5 < global_centroid_list[i] < Tx + 2.5:
                #         flag_x = True
                #     if Ty - 2.5 < global_centroid_list[i + 1] < Ty + 2.5:
                #         flag_y = True

                # if not flag_x or not flag_y:
                
                return (Tx, Ty)

        except Exception as e:
            print(f"Error in global_transformation: {e}")

        return None

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
        # Initialize the ROS node
        rospy.init_node('human_detection_node', anonymous=True)
        node = HumanDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass