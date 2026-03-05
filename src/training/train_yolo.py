import os
from ultralytics import YOLOWorld

def train_yolo_model(data_yaml_path, epochs, output_dir="models/checkpoints/trained/yolo"):
    print("[*] Fine-tuning YOLO locally...")
    model = YOLOWorld('yolov8s-world.pt')
    results = model.train(data=data_yaml_path, epochs=epochs, imgsz=640, project=output_dir, name="run")
    os.rename(os.path.join(results.save_dir, "weights/best.pt"), os.path.join(output_dir, "best_yolo.pt"))