import cv2
import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/yolov5s.pt')
model.classes = [0]  # Detect only person

# Image
im1 = Image.open('src/image.jpeg')  # PIL image

# Inference
results = model([im1], size=640)

# Extract bounding box coordinates for person class
predictions = results.pandas().xyxy[0]  # Get predictions as a pandas DataFrame
person_predictions = predictions[predictions['name'] == 'person']  # Filter person class predictions

# Iterate over person predictions and extract bounding box coordinates
bounding_boxes = []
for _, row in person_predictions.iterrows():
    xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
    bounding_box = (xmin, ymin, xmax, ymax)
    bounding_boxes.append(bounding_box)

# Print the bounding box coordinates
for bbox in bounding_boxes:
    print(f"Bounding Box: {bbox}")