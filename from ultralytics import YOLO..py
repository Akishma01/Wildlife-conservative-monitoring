from ultralytics import YOLO
import cv2

# Load YOLOv5 pretrained model
model = YOLO("yolov5s.pt")  # Downloads automatically

# Load image
image_path = "wildlife.jpeg"
results = model(image_path)

# Plot results
results[0].show()  # Shows bounding boxes

# Or save output
results[0].save(filename="wildlife.jpeg")