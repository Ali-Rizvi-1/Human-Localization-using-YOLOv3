# Human Detection and Localization using YOLO

This repository contains the code for the "Human Detection and Localization" module. This code implements YOLOv5 for human detection, performs local transformation of human coordinates using a perspective projection model, and then transforms the local coordinates into global coordinates using pose information from RTAB-Map (SLAM algorithm).

## Prerequisites

Before running the code, please ensure you have the following installed:

- Python (version 3.6 or higher)
- PyTorch
- OpenCV
- PIL (Python Imaging Library)

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/Ali-Rizvi-1/Human-Localization-using-YOLOv3.git

2. Install the required dependencies:

   ```shell
   pip install -r requirements.txt

3. Download the YOLOv5 model weights from the following link: YOLOv5 Weights. Save the weights file to the weights/ directory in the repository.

## Usage

1. Ensure that you have an image (image.jpeg) in the src/ directory on which you want to perform human detection.4

2. Open the main.py file in a text editor and update the weights_path variable with the correct path to the YOLOv5 weights file (weights/yolov5s.pt).

3. Run the main.py script:

   ```shell
   python3 main.py
   
This will execute the code and perform human detection on the specified image. The bounding box coordinates of detected humans will be printed to the console.

4. Check the console output for the bounding box coordinates of detected humans.

## Example Output

1. output:

   ```shell
   Bounding Box: (x1, y1, x2, y2)
   Bounding Box: (x1, y1, x2, y2)

Please make sure to provide the correct path to the YOLOv5 weights file and ensure that the image file (image.jpeg) is present in the src/ directory before running the code.

For more details on using torch.hub.load, refer to the PyTorch Hub Model Loading documentation.

## System Diagram

![SystemDiagram](https://github.com/Ali-Rizvi-1/Human-Localization-using-YOLOv3/blob/main/Capstone%20-%20SystemDiagram.png)

By following these instructions, you should be able to run and execute the code successfully.