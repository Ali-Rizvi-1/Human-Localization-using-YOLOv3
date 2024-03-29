# Human Detection and Localization using Yolo
This repository contains the code for the "Human Detection and Localization" module. This code implements YOLOv3 for the human detection, then uses perspective projection model for the local transformation of human coordinates, and finally, using the pose information from RTAB-Map (SLAM algorithm) it transforms the local coordinates into global coordinates.  

The weights used for the YOLOv3 model can be found on the following link: https://bit.ly/3pqtJUT

Below is the system level diagram of the project, and it shows the connection of individual modules in the overall pipeline of the project.

![SystemDiagram](https://github.com/Ali-Rizvi-1/Human-Localization-using-YOLOv3/blob/main/Capstone%20-%20SystemDiagram.png)
