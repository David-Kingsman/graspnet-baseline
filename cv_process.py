'''
This script is used to segment an object from an image using SAM and YOLO-World.
It will detect the object using YOLO-World and then use SAM to segment the object.
It will save the segmentation mask to a file.

bash command:
python cv_process.py captures/20251020_143527_956  
'''

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor


def choose_model():
    """Initialize SAM predictor with proper parameters for segmentation"""
    model_weight = 'logs/models/sam_b.pt'
    overrides = dict(
        task='segment',
        mode='predict',
        imgsz=1024,
        model=model_weight,
        conf=0.25,
        save=False,
        device='cpu'  # use CPU to avoid GPU memory shortage
    )
    return SAMPredictor(overrides=overrides)


def set_classes(model, target_class):
    """Set YOLO-World model to detect specific class for object detection"""
    model.set_classes([target_class])


def detect_objects(image_path, target_class=None):
    """
    Detect objects with YOLO-World for object detection
    Returns: (list of bboxes in xyxy format, detected classes list, visualization image)
    """
    model = YOLO("logs/models/yolov8s-world.pt")
    if target_class:
        set_classes(model, target_class)

    results = model.predict(image_path)
    boxes = results[0].boxes
    vis_img = results[0].plot()  # Get visualized detection results

    # Extract valid detections
    valid_boxes = []
    for box in boxes:
        if box.conf.item() > 0.25:  # Confidence threshold
            valid_boxes.append({
                "xyxy": box.xyxy[0].tolist(),
                "conf": box.conf.item(),
                "cls": results[0].names[box.cls.item()]
            })

    return valid_boxes, vis_img


def process_sam_results(results):
    """Process SAM results to get mask and center point for segmentation"""
    if not results or not results[0].masks:
        return None, None

    # Get first mask (assuming single object segmentation)
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255

    # Find contour and center
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    M = cv2.moments(contours[0])
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


def segment_image(image_path, output_mask='mask1.png'):
    """Segment an image using SAM and YOLO-World for object segmentation"""
    # User input for target class
    use_target_class = input("Detect specific class? (yes/no): ").lower() == 'yes'
    target_class = input("Enter class name: ").strip() if use_target_class else None

    # Detect objects
    detections, vis_img = detect_objects(image_path, target_class)
    cv2.imwrite('detection_visualization.jpg', vis_img)

    # Prepare SAM predictor
    predictor = choose_model()
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    predictor.set_image(image)  # Set image for SAM

    if detections:
        # Show all detections and let user choose
        print(f"\detected objects:")
        for i, det in enumerate(detections):
            print(f"{i+1}. {det['cls']} (confidence: {det['conf']:.2f})")
        
        if len(detections) > 1:
            try:
                choice = int(input(f"select the object to segment (1-{len(detections)}): ")) - 1
                if 0 <= choice < len(detections):
                    selected_det = detections[choice]
                else:
                    selected_det = detections[0]  # default select the first one
            except:
                selected_det = detections[0]  # default select the first one
        else:
            selected_det = detections[0]
        
        results = predictor(bboxes=[selected_det["xyxy"]])
        center, mask = process_sam_results(results)
        print(f"selected {selected_det['cls']} (confidence: {selected_det['conf']:.2f})")
    else:
        # Manual point selection
        print("No detections - click on target object")
        cv2.imshow('Select Object', vis_img)
        point = []

        def click_handler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                point.extend([x, y])
                cv2.destroyAllWindows()

        cv2.setMouseCallback('Select Object', click_handler)
        cv2.waitKey(0)

        if len(point) == 2:
            results = predictor(points=[point], labels=[1])
            center, mask = process_sam_results(results)
        else:
            raise ValueError("No selection made")

    # Save results
    if mask is not None:
        cv2.imwrite(output_mask, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
        print(f"Segmentation saved to {output_mask}")
    else:
        print("mask1")

    return mask


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        # if provided parameter, use specified directory
        data_dir = sys.argv[1]
        # automatically find color file
        import os
        color_files = [f for f in os.listdir(data_dir) if f.endswith('_color.png') or f == 'color.png']
        if color_files:
            image_path = os.path.join(data_dir, color_files[0])
            output_mask = os.path.join(data_dir, 'mask1.png')
            print(f"Using image file: {image_path}")
            print(f"Saving mask to: {output_mask}")
            segment_image(image_path, output_mask)
        else:
            print(f"No *_color.png file found in {data_dir}")
    else:
        # default use color.png in current directory
        segment_image('color.png')