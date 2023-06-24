from typing import List, Tuple
import cv2
import torch
from PIL import Image

class PersonDetector:
    def __init__(self, weights_path: str) -> None:
        self.model = self._load_model(weights_path)

    def _load_model(self, path: str) -> torch.nn.Module:
        """
        Load pretrained yolov5s model and detect only person.

        Args:
            path (str): Path to the model weights file.

        Returns:
            torch.nn.Module: Loaded YOLOv5 model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)
        model.classes = [0]  # Detect only person
        return model

    def detect_person(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Detects people in the image and returns the bounding box coordinates of humans.

        Args:
            image_path (str): Path to the image file.

        Returns:
            List[Tuple[int, int, int, int]]: List of bounding box coordinates (xmin, ymin, xmax, ymax).
        """
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Perform inference
        results = self.model([image], size=640)

        # Extract bounding box coordinates for person class
        predictions = results.pandas().xyxy[0]  # Get predictions as a pandas DataFrame
        person_predictions = predictions[predictions['name'] == 'person']  # Filter person class predictions

        # Extract bounding box coordinates
        bounding_boxes = []
        for _, row in person_predictions.iterrows():
            xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
            bounding_box = (int(xmin), int(ymin), int(xmax), int(ymax))
            bounding_boxes.append(bounding_box)

        # Print bounding box coordinates
        for bbox in bounding_boxes:
            print(f"Bounding Box: {bbox}")

        return bounding_boxes