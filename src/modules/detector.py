from ultralytics import YOLOWorld


class ObjectDetector:
    def __init__(self, model_path, conf_threshold):
        self.model = YOLOWorld(model_path)
        self.conf = conf_threshold

    def detect(self, image_path, prompt):
        self.model.set_classes([prompt])
        results = self.model.predict(image_path, conf=self.conf, verbose=False)
        if not results or len(results[0].boxes) == 0:
            raise ValueError(f"No object found for text prompt: {prompt}")
        # Explicitly select box with highest confidence
        boxes = results[0].boxes
        best_idx = boxes.conf.argmax()
        return boxes[best_idx].xyxy.cpu().numpy()[0]
