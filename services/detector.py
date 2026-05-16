from PIL import Image
from io import BytesIO

def detect_objects(model, image_bytes: bytes) -> tuple:

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_height = image.size[1]

        results = model(image,verbose=False,conf=0.45, iou=0.45)

        detections = []
        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls)]
                confidence = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_height = y2 - y1

                detections.append({
                    "label": label,
                    "confidence": round(confidence, 2),
                    "bbox_height": round(bbox_height, 1),
                    "bbox": {          # pour le CNN crop
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                })
        return detections, image_height
    except Exception as e:
        raise ValueError(f"Erreur analyse image : {e}")
