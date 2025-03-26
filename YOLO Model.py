%pip install ultralytics
import ultralytics
ultralytics.checks()

# Train YOLO11n on COCO8 for 3 epochs
!yolo train model=yolo11n.pt data=/content/Dataset/data.yaml epochs=10 batch=2 imgsz=640

# Run inference on an image with YOLO11n
!yolo predict model=/content/runs/detect/train/weights/best.pt source='/content/Dataset/test/images/input_image_active-5-_png.rf.487977afebf71480b00c1a6ad06c8c82.jpg'