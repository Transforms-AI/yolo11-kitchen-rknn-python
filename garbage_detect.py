import os
import cv2
import json
import logging
import torch
import sys
import argparse
import threading
from queue import Queue
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

# --- Logger setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# --- Class ID and Label Constants ---
PERSON_CLASS_ID_WORLD = 0           # Usually 'person' class
YOLO_GARBAGE_ID = 1                 # ID for YOLO garbage
YOLOE_GARBAGE_ID = 2                # ID for YOLOE garbage

YOLO_GARBAGE_NAME = "garbage-yolo"
YOLOE_GARBAGE_NAME = "garbage-yoloe"

# --- Load configuration from JSON ---
def load_config(config_path):
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}", exc_info=True)
        sys.exit(1)

# --- Load YOLO model from directory path ---
def load_yolo_model(model_dir):
    try:
        model = YOLO(model_dir)
        logger.info(f"Model loaded from: {model_dir}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_dir}: {e}", exc_info=True)
        sys.exit(1)

# --- Run inference and put result in a queue ---
def run_inference(model, frame, results_queue, conf=0.25):
    try:
        results = model.predict(frame, verbose=False, conf=conf)
        results_queue.put(results[0] if results else None)
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        results_queue.put(None)

# --- IoU calculation helpers ---
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_iou_components(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou, interArea, boxAArea, boxBArea

# --- Combine or filter results from both models ---
def filter_and_combine_results(results_world, results_yolo, person_iou_thresh, overlap_iou_thresh):
    if results_world is None or results_yolo is None:
        logger.warning("Invalid input results for filtering. One or both results are None.")
        return None

    processing_device = 'cpu'
    boxes_world_data = results_world.boxes.data.to(processing_device) if results_world.boxes is not None else torch.empty((0, 6))
    boxes_yolo_data = results_yolo.boxes.data.to(processing_device) if results_yolo.boxes is not None else torch.empty((0, 6))

    original_world_indices = list(range(len(boxes_world_data)))
    original_yolo_indices = list(range(len(boxes_yolo_data)))

    removed_world_indices_step1 = set()
    removed_yolo_indices_step1 = set()

    person_indices_world = [i for i in original_world_indices if int(boxes_world_data[i][5]) == PERSON_CLASS_ID_WORLD]

    for i_person in person_indices_world:
        box_person = boxes_world_data[i_person][:4].cpu().numpy()

        for j_world in original_world_indices:
            if j_world == i_person or int(boxes_world_data[j_world][5]) == PERSON_CLASS_ID_WORLD:
                continue
            if j_world in removed_world_indices_step1:
                continue
            box_candidate = boxes_world_data[j_world][:4].cpu().numpy()
            _, inter_area, _, bc_area = calculate_iou_components(box_person, box_candidate)
            iog = inter_area / bc_area if bc_area > 0 else 0.0
            if iog > person_iou_thresh:
                removed_world_indices_step1.add(j_world)

        for j_yolo in original_yolo_indices:
            if j_yolo in removed_yolo_indices_step1:
                continue
            box_candidate = boxes_yolo_data[j_yolo][:4].cpu().numpy()
            _, inter_area, _, bc_area = calculate_iou_components(box_person, box_candidate)
            iog = inter_area / bc_area if bc_area > 0 else 0.0
            if iog > person_iou_thresh:
                removed_yolo_indices_step1.add(j_yolo)

    kept_world_indices_step1 = [i for i in original_world_indices if i not in removed_world_indices_step1]
    kept_yolo_indices_step1 = [i for i in original_yolo_indices if i not in removed_yolo_indices_step1]

    removed_yolo_indices_step2 = set()
    world_garbage_indices_for_step2 = [i for i in kept_world_indices_step1 if int(boxes_world_data[i][5]) != PERSON_CLASS_ID_WORLD]

    for i_world in world_garbage_indices_for_step2:
        box_world = boxes_world_data[i_world][:4].cpu().numpy()
        for j_yolo in kept_yolo_indices_step1:
            if j_yolo in removed_yolo_indices_step2:
                continue
            box_yolo = boxes_yolo_data[j_yolo][:4].cpu().numpy()
            iou_val = calculate_iou(box_world, box_yolo)
            if iou_val > overlap_iou_thresh:
                removed_yolo_indices_step2.add(j_yolo)

    final_kept_yolo_indices = [i for i in kept_yolo_indices_step1 if i not in removed_yolo_indices_step2]
    final_kept_world_indices = [i for i in kept_world_indices_step1 if int(boxes_world_data[i][5]) != PERSON_CLASS_ID_WORLD]

    final_boxes_world_tensor = boxes_world_data[final_kept_world_indices] if final_kept_world_indices else torch.empty((0, 6))
    final_boxes_yolo_tensor = boxes_yolo_data[final_kept_yolo_indices] if final_kept_yolo_indices else torch.empty((0, 6))

    if len(final_boxes_yolo_tensor) > 0:
        final_boxes_yolo_tensor[:, 5] = YOLO_GARBAGE_ID
    if len(final_boxes_world_tensor) > 0:
        final_boxes_world_tensor[:, 5] = YOLOE_GARBAGE_ID

    combined_boxes_data = torch.cat((final_boxes_yolo_tensor, final_boxes_world_tensor), dim=0)
    combined_boxes = Boxes(combined_boxes_data, results_world.orig_shape) if len(combined_boxes_data) > 0 else None

    combined_results = results_world.new()
    combined_results.boxes = combined_boxes
    combined_results.names = {
        YOLO_GARBAGE_ID: YOLO_GARBAGE_NAME,
        YOLOE_GARBAGE_ID: YOLOE_GARBAGE_NAME
    }
    return combined_results

# --- Main Detector Class ---
class DualModelDetector:
    def __init__(self, model_yoloe, model_yolo, config):
        self.model_yoloe = model_yoloe
        self.model_yolo = model_yolo
        self.config = config
    

    def detect(self, frame):
        
        person_iou_threshold = self.config.get("person_iou_threshold", 0.3)
        overlap_iou_threshold = self.config.get("overlap_iou_threshold", 0.7)

        q_yoloe = Queue()
        q_yolo = Queue()

        t1 = threading.Thread(target=run_inference, args=(self.model_yoloe, frame.copy(), q_yoloe), daemon=True)
        t2 = threading.Thread(target=run_inference, args=(self.model_yolo, frame.copy(), q_yolo, 0.25), daemon=True)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        results_yoloe = q_yoloe.get()
        results_yolo = q_yolo.get()

        if results_yoloe is None or results_yolo is None:
            return None

        return filter_and_combine_results(results_yoloe, results_yolo, person_iou_threshold, overlap_iou_threshold)

# --- Main CLI Application ---
def main(source_img):
    config = load_config("config.json")
    model_yoloe = load_yolo_model(config["yoloe"])
    model_yolo = load_yolo_model(config["yolo"])

    detector = DualModelDetector(model_yoloe, model_yolo, config)
    frame = cv2.imread(source_img)
    if frame is None:
        print(f" Failed to read image: {source_img}")
        return 1

    results = detector.detect(frame)
    if results is None or results.boxes is None:
        print(" No detections found or inference failed.")
        return 1

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    print(f"\n Detected {len(boxes)} object(s):")
    for i, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = map(int, box[:4])
        conf = float(confs[i - 1])
        cls = int(classes[i - 1])
        print(f"  Box {i}: ({x1}, {y1}) -> ({x2}, {y2}), Class: {cls}, Confidence: {conf:.2f}")

    return 0

# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual YOLO Inference")
    parser.add_argument("--source", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    exit_code = main(args.source)
    sys.exit(exit_code)
