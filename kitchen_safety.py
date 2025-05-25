import cv2
from libraries.datasend import DataUploader
from libraries.utils import time_to_string, mat_to_response
from libraries.drawing import draw_boxes
from libraries.stream_publisher import StreamPublisher
from libraries.async_capture import VideoCaptureAsync
import json
import time
import torch
from ultralytics import YOLO 
import logging # Import logging

# --- Global Logger ---
logger = logging.getLogger("kitchen_safety")

def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: A list or tuple representing the first box in the format [x1, y1, x2, y2].
        box2: A list or tuple representing the second box in the format [x1, y1, x2, y2].

    Returns:
        The IoU value (float) between 0 and 1.
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou

def live(model: YOLO, config, names, person_model: YOLO, device: str):
    """
    Performs object detection on a live video stream from a camera using Ultralytics YOLO,
    sends data to the server when specific classes are detected,
    and incorporates person-based grounding for kitchen safety detections.

    Args:
        model: The Ultralytics YOLO model for kitchen safety.
        config: Configuration dictionary loaded from config.json.
        names: Dictionary mapping class IDs to class names for the main model.
        person_model: The Ultralytics YOLO model for person detection.
        device: The device to run inference on ('cpu' or 'cuda').
    """

    # Setup video
    if config['local_video']:
        video_source = config["local_video_source"]
        cap_async = VideoCaptureAsync(video_source, loop=True)
    else:
        video_source = config['video_source']
        cap_async = VideoCaptureAsync(video_source)
        
    cap_async.start()

    # Setup streaming
    streamer = None
    if config['livestream']:
        streamer = StreamPublisher(
            "live_" + config['sn'], start_stream=False, host=config['local_ip'], port=1883, jpeg_quality=70, target_width=1600
        )
        streamer.start_streaming()
        logger.debug(f"Streamer initialized for topic: live_{config['sn']}")

    # Initialize counter and timers
    frame_count = 0
    last_datasend_time = time.time()
    last_heartbeat_time = time.time()
    last_inference_time = time.time()

    # Initialize DataUploader
    api_url = config['data_send_url']
    heartbeat_url = config['heartbeat_url']
    headers = {"X-Secret-Key": config["X-Secret-Key"]}
    data_uploader = DataUploader(api_url, heartbeat_url, headers)
    
    # Model inference parameters
    main_model_conf = 0.3
    main_model_iou = 0.2
    person_model_conf = 0.2
    person_model_iou = 0.2
    use_half = (device == 'cuda')

    try:
        while True:
            ret, frame = cap_async.read()
            
            if not ret:
                logger.error("Could not read frame.")
                if config.get('local_video', False) and cap_async.loop:
                    logger.debug("Local video loop: restarting.")
                    continue
                else: 
                    time.sleep(0.5) 
                    if not cap_async.isOpened(): 
                        logger.warning("Stream closed. Attempting to reconnect...")
                        cap_async.release()
                        cap_async = VideoCaptureAsync(video_source, loop=config.get('local_video', False) and cap_async.loop)
                        cap_async.start()
                        if not cap_async.isOpened():
                            logger.error("Reconnect failed. Exiting.")
                            break
                        else:
                            logger.info("Reconnected successfully.")
                            continue
                    continue

            frame_count += 1
            current_time = time.time()
            # Send Heartbeat
            if current_time - last_heartbeat_time >= config["heartbeat_interval"]:
                data_uploader.send_heartbeat(
                    config["sn"], config["local_ip"], time_to_string(current_time)
                )
                last_heartbeat_time = current_time
                logger.debug(f"Heartbeat sent at {time_to_string(current_time)}")
                
            # Perform inference every 'inference_interval' seconds
            if current_time - last_inference_time >= config["inference_interval"]:
                logger.debug(f"Performing inference for frame {frame_count} at {time_to_string(current_time)}")

                # --- Person Detection ---
                person_results_obj = person_model.predict(frame, conf=person_model_conf, iou=person_model_iou, device=device, half=use_half, verbose=False)
                if person_results_obj and len(person_results_obj) > 0 and person_results_obj[0].boxes is not None and len(person_results_obj[0].boxes.xyxy) > 0:
                    logger.debug(f"Person model: Found {len(person_results_obj[0].boxes.xyxy)} raw detections.")
                else:
                    logger.debug("Person model: No detections found.")
                
                temp_person_boxes = []
                temp_person_class_ids = []

                if person_results_obj and len(person_results_obj) > 0 and person_results_obj[0].boxes and len(person_results_obj[0].boxes.xyxy) > 0:
                    temp_person_boxes = person_results_obj[0].boxes.xyxy.cpu().numpy()
                    temp_person_class_ids = person_results_obj[0].boxes.cls.cpu().numpy().astype(int)
                
                person_boxes = []
                for i, box_coords in enumerate(temp_person_boxes):
                    if temp_person_class_ids[i] == 0: # Assuming 'person' is class_id 0 for the person_model
                        person_boxes.append(box_coords)
                
                if len(temp_person_boxes) > 0: # Log only if there were initial detections
                     logger.debug(f"Person model: Extracted {len(person_boxes)} person boxes (class 0) from {len(temp_person_boxes)} raw detections.")

                # --- Main Model Inference (Kitchen Safety) ---
                results_obj = model.predict(frame, conf=main_model_conf, iou=main_model_iou, device=device, half=use_half, verbose=False)
                if results_obj and len(results_obj) > 0 and results_obj[0].boxes is not None and len(results_obj[0].boxes.xyxy) > 0:
                    logger.debug(f"Main model: Found {len(results_obj[0].boxes.xyxy)} raw detections.")
                else:
                    logger.debug("Main model: No detections found.")

                boxes = []
                class_ids = []
                scores = []

                if results_obj and len(results_obj) > 0 and results_obj[0].boxes and len(results_obj[0].boxes.xyxy) > 0:
                    boxes = results_obj[0].boxes.xyxy.cpu().numpy()
                    class_ids = results_obj[0].boxes.cls.cpu().numpy().astype(int)
                    scores = results_obj[0].boxes.conf.cpu().numpy()

                if len(boxes) > 0: # Log only if there were initial detections
                    logger.debug(f"Main model: Extracted {len(boxes)} total boxes after initial processing.")

                # --- Grounding with Person Detections ---
                filtered_boxes = []
                filtered_class_ids = []
                filtered_scores = []

                person_related_classes = [0, 1, 2, 3, 4, 5, 8, 10]  # Classes related to a person
                iou_threshold = config['iou_threshold'] 

                logger.debug(f"Grounding: Number of main model boxes before grounding: {len(boxes)}")
                logger.debug(f"Grounding: Number of person boxes for grounding: {len(person_boxes)}")
                logger.debug(f"Grounding: Person-related classes: {person_related_classes}, IoU threshold: {iou_threshold}")

                for i, box_coords in enumerate(boxes):
                    class_id = class_ids[i]
                    score = scores[i]

                    if class_id not in person_related_classes:
                        filtered_boxes.append(box_coords)
                        filtered_class_ids.append(class_id)
                        filtered_scores.append(score)
                        logger.debug(f"Grounding: Kept non-person-related class {names.get(class_id, class_id)}")
                    else:
                        grounded = False
                        for person_box_coords in person_boxes:
                            iou = calculate_iou(box_coords, person_box_coords)
                            if iou > iou_threshold:
                                grounded = True
                                logger.debug(f"Grounding: Class {names.get(class_id, class_id)} grounded with person (IoU: {iou:.2f})")
                                break
                        
                        if grounded:
                            filtered_boxes.append(box_coords)
                            filtered_class_ids.append(class_id)
                            filtered_scores.append(score)
                        else: # Only log if it was discarded
                            logger.debug(f"Grounding: Discarded person-related class {names.get(class_id, class_id)} (no sufficient IoU with any person)")
                
                logger.debug(f"Grounding: Number of main model boxes after grounding: {len(filtered_boxes)}")
                if len(filtered_boxes) > 0:
                    filtered_names = [names.get(cid, cid) for cid in filtered_class_ids]
                    logger.debug(f"Grounding: Filtered class names after grounding: {filtered_names}")

                inferred_classes = [names[class_id] for class_id in filtered_class_ids if class_id in names] 
                logger.info(f"Frame {frame_count}: Inferred classes - {inferred_classes}")

                violation_classes = [1, 3, 5, 6, 9, 10] 
                violation_list = []
                violation_class_ids = []  
                violation_boxes = [] 

                for i, class_id in enumerate(filtered_class_ids):
                    if class_id in violation_classes and class_id in names: 
                        violation_list.append(names[class_id])
                        violation_class_ids.append(class_id)
                        violation_boxes.append(filtered_boxes[i])
                
                logger.debug(f"Violations found: {list(set(violation_list))}")

                display_frame = frame.copy()
                if config['draw']:
                    display_frame = draw_boxes(display_frame, violation_boxes, violation_class_ids, names)
                
                if time.time() - last_datasend_time >= config['datasend_interval']:
                    start_time_str = time_to_string(last_datasend_time)
                    end_time_str = time_to_string(current_time)
                    
                    data = {
                        "sn": config['sn'],
                        "violation_list": json.dumps(list(set(violation_list))),
                        "violation": True if len(violation_list) != 0 else False,
                        "start_time": start_time_str,
                        "end_time": end_time_str
                    }
                    files = {"image": mat_to_response(display_frame)}

                    if config.get('send_data', True):
                        data_uploader.send_data(data, files=files)
                        logger.debug(f"Data sent. Payload: {data}")
                    else: 
                        logger.debug(f"Data sending skipped (send_data is False). Would have sent: {data}")
                    
                    last_datasend_time = time.time()
                
                if config['livestream'] and streamer:
                    streamer.updateFrame(display_frame)
                    logger.debug("Livestream frame updated.")
                
                if config["show"]:
                    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
                    cv2.imshow("Output", display_frame)
                    key = cv2.waitKey(1)
                    if key == 27: # ESC key
                        logger.info("ESC key pressed, exiting.")
                        break
                    
                last_inference_time = current_time
    finally:
        cap_async.release()
        cv2.destroyAllWindows()        
        if config['livestream'] and streamer:
            streamer.stop_streaming()
            logger.debug("Streamer stopped.")
        logger.info("Application cleanup finished.")

# Usage
if __name__ == '__main__':
    # Load configuration from config.json
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        # Basic logger setup for this critical error
        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.error("config.json not found. Please ensure it exists in the current directory.")
        exit(1)
    except json.JSONDecodeError:
        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.error("Error decoding config.json. Please check its format.")
        exit(1)
    
    # --- Logger Setup ---
    log_level_str = config.get('logging_level', 'INFO').upper()
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = log_level_map.get(log_level_str, logging.INFO)
    logger.setLevel(log_level)

    # Create console handler
    ch = logging.StreamHandler()
    # Formatter can be created once and used for all handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # Add handler to the logger, prevent adding multiple handlers if re-run
    if not logger.handlers:
        logger.addHandler(ch)

    logger.info(f"Logging level set to {log_level_str}.")
    logger.debug(f"Full config: {config}") # Log full config only at DEBUG level
        
    # label mapping from array to dict
    labels = ['hat','no_hats', 'mask','no_masks','gloves','no_gloves','food_uncovered','pilgrim','no_pilgrim','garbage','incorrect_mask','food_processing']
    names = {}
    for i, label in enumerate(labels):
        names[i] = label
    logger.debug(f"Class names mapping: {names}")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}") # Info level for device is good
        
    # Load the kitchen safety model using Ultralytics YOLO
    model_path = config["model"]
    logger.debug(f"Attempting to load Kitchen safety model: {model_path} on device: {device}")
    try:
        model = YOLO(model_path, task='detect')
        logger.info(f"Kitchen safety model '{model_path}' loaded on {device}.")
        if model: # Check if model loaded
            model_info = {
                "type": type(model).__name__, # Get class name as string
                "device": str(model.device), # Convert device to string
                "names": model.names if hasattr(model, 'names') else 'N/A',
            }
            logger.debug(f"Kitchen safety model details: {model_info}")
    except Exception as e:
        logger.critical(f"Failed to load Kitchen safety model from {model_path}: {e}", exc_info=True)
        exit(1)
        
    person_model_path = config['person_model']
    logger.debug(f"Attempting to load Person detection model: {person_model_path} on device: {device}")
    try:
        person_model = YOLO(person_model_path, task='detect')
        logger.info(f"Person detection model '{person_model_path}' loaded on {device}.")
        if person_model: # Check if model loaded
            person_model_info = {
                "type": type(person_model).__name__,
                "device": str(person_model.device),
                "names": person_model.names if hasattr(person_model, 'names') else 'N/A',
            }
            logger.debug(f"Person detection model details: {person_model_info}")
    except Exception as e:
        logger.critical(f"Failed to load Person detection model from {person_model_path}: {e}", exc_info=True)
        exit(1)

    try:
        live(model, config, names, person_model, device)
    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C). Exiting.")
    except Exception as e:
        logger.critical(f"An unhandled exception occurred in the live loop: {e}", exc_info=True)
    finally:
        logger.info("Application shutting down.")