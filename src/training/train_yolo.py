import os
from ultralytics import YOLOWorld

def train_yolo_model(data_yaml_path, epochs, output_dir="models/checkpoints/trained/yolo"):
    print("[*] Fine-tuning YOLO locally...")
    model = YOLOWorld('yolov8s-world.pt')
    results = model.train(data=data_yaml_path, epochs=epochs, imgsz=640, project=output_dir, name="run")
    
    # YOLO returns a path in results.save_dir directly when project and name are specified.
    # It seems 'results' is a DetMetrics object which might not have save_dir.
    # We should get the save_dir from the model trainer or just use the known path.
    # Actually, model.train() returns a DetMetrics object but 'results.save_dir' might not be correct or missing 'weights/best.pt'.
    # Specifically, Yolov8 saves to project/name/weights/best.pt
    
    # We need to construct the path manually based on the trainer's save_dir OR use the trainer's save_dir if available.
    if hasattr(os, 'makedirs'):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        best_pt_path = os.path.join(results.save_dir, "weights/best.pt")
        os.rename(best_pt_path, os.path.join(output_dir, "best_yolo.pt"))
    except Exception as e:
        print(f"Warning: Could not rename from results.save_dir. Trying to find best.pt manually. Error: {e}")
        # Find the latest run folder
        import glob
        run_dirs = sorted(glob.glob(os.path.join(output_dir, "run*")))
        if run_dirs:
            latest_run = run_dirs[-1]
            best_pt_path = os.path.join(latest_run, "weights/best.pt")
            if os.path.exists(best_pt_path):
                import shutil
                shutil.copy(best_pt_path, os.path.join(output_dir, "best_yolo.pt"))
                print(f"Copied {best_pt_path} to {os.path.join(output_dir, 'best_yolo.pt')}")
            else:
                print(f"Error: {best_pt_path} not found.")
        else:
            print(f"Error: No run directories found in {output_dir}")
