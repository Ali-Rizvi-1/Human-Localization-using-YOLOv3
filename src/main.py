import time
import cv2
from PIL import Image
from person_detector import PersonDetector

detection_model = PersonDetector('weights/yolov5s.pt')

image_path = "src/image.jpeg"
# image = cv2.imread(image_path)
image = Image.open(image_path)
detection_model.detect_person(image)