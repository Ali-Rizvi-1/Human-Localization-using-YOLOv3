from person_detector import PersonDetector

detection_model = PersonDetector('weights/yolov5s.pt')
detection_model.detect_person("src/image.jpeg")