import cv2
from libraries.datasend import DataUploader
from libraries.utils import time_to_string, mat_to_response
from libraries.drawing import draw_boxes
from libraries.stream_publisher import StreamPublisher
from libraries.stream_receiver import StreamReceiver
from libraries.async_capture import VideoCaptureAsync
import json
import time
import os
import torch
from libraries.rknn import RKNN_instance
    
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

def demo(model: RKNN_instance, config, names, person_model):
    """
    Performs object detection on a video stream using either RKNN or YOLO (Ultralytics),
    with person-based grounding for kitchen safety detections.

    Args:
        model: The RKNN model instance or YOLO model.
        config: Configuration dictionary loaded from config.json.
        names: Dictionary mapping class IDs to class names (used for RKNN).
        person_model: The YOLO model for person detection.
    """
    
    # Setup video
    video_source = config["local_video_source"]
    cap_async = VideoCaptureAsync(video_source, loop=True)
    cap_async.start()

    cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap_async.read()

        if not ret:
            print("End of video stream or error reading frame.")
            continue

        # Detect objects
        start = time.perf_counter()

        # --- Person Detection ---
        person_results = person_model.predict(frame, half=True)
        person_result = person_results[0]
        person_boxes = person_result.boxes.xyxy.cpu().numpy()
        person_class_ids = person_result.boxes.cls.cpu().numpy().astype(int)

        # Filter for person class (assuming class ID 0 represents 'person')
        person_boxes = [box for i, box in enumerate(person_boxes) if person_class_ids[i] == 0]
        
        # RKNN inference
        boxes, class_ids, scores = model.detect(frame)
        inference_time = (time.perf_counter() - start) * 1000
        print(f"Inference time (RKNN): {inference_time:.2f} ms")

        # --- Grounding with Person Detections ---
        filtered_boxes = []
        filtered_class_ids = []
        filtered_scores = []

        person_related_classes = [0, 1, 2, 3, 4, 5, 8, 10]  # Classes related to a person (hat, no_hat, mask, etc.)
        # person_related_classes = []
        iou_threshold = config['iou_threshold']  # Adjust as needed

        for i, box in enumerate(boxes):
            class_id = class_ids[i]
            score = scores[i]

            if class_id not in person_related_classes:
                # Keep non-person-related detections as is
                filtered_boxes.append(box)
                filtered_class_ids.append(class_id)
                filtered_scores.append(score)
            else:
                # Check for overlap with person boxes
                grounded = False
                for person_box in person_boxes:
                    iou = calculate_iou(box, person_box)
                    if iou > iou_threshold:
                        grounded = True
                        break

                if grounded:
                    filtered_boxes.append(box)
                    filtered_class_ids.append(class_id)
                    filtered_scores.append(score)

        # Draw detections on the frame
        combined_img = draw_boxes(frame.copy(), filtered_boxes, filtered_class_ids, names)

        cv2.imshow("Output", combined_img)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap_async.release()
    cv2.destroyAllWindows()

def live(model: RKNN_instance, config, names, person_model : RKNN_instance):
    """
    Performs object detection on a live video stream from a camera,
    sends data to the server when specific classes are detected,
    and incorporates person-based grounding for kitchen safety detections.

    Args:
        model: The RKNN model instance or YOLO model.
        config: Configuration dictionary loaded from config.json.
        names: Dictionary mapping class IDs to class names (used for RKNN).
        person_model: The YOLO model for person detection.
    """
    # Setup video
    video_source = config["video_source"]
    cap_async = VideoCaptureAsync(video_source)
    cap_async.start()

    # Setup streaming
    if config['livestream']:
        streamer = StreamPublisher(
            "live_" + config['sn'], start_stream=False, host=config['local_ip'], port=1883
        )
        streamer.start_streaming()

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
    
    try:
        while True:
            ret, frame = cap_async.read()
            
            if not ret:
                print("Error: Could not read frame.")
                continue

            frame_count += 1
            current_time = time.time()
            # Send Heartbeat
            if current_time - last_heartbeat_time >= config["heartbeat_interval"]:
                data_uploader.send_heartbeat(
                    config["sn"], config["local_ip"], time_to_string(current_time)
                )
                last_heartbeat_time = current_time
                
            # Perform inference every 'inference_interval' seconds
            if current_time - last_inference_time >= config["inference_interval"]:
                
                # --- Person Detection ---
                # person_results = person_model.predict(frame, half=True)
                # person_result = person_results[0]
                # person_boxes = person_result.boxes.xyxy.cpu().numpy()
                # person_class_ids = person_result.boxes.cls.cpu().numpy().astype(int)
                person_boxes, person_class_ids, _ = person_model.detect(frame)

                # Filter for person class (assuming class ID 0 represents 'person')
                person_boxes = [box for i, box in enumerate(person_boxes) if person_class_ids[i] == 0]

                # RKNN inference
                boxes, class_ids, scores = model.detect(frame)

                # --- Grounding with Person Detections ---
                filtered_boxes = []
                filtered_class_ids = []
                filtered_scores = []

                person_related_classes = [0, 1, 2, 3, 4, 5, 8, 10]  # Classes related to a person
                # person_related_classes = []
                iou_threshold = config['iou_threshold'] 

                for i, box in enumerate(boxes):
                    class_id = class_ids[i]
                    score = scores[i]

                    if class_id not in person_related_classes:
                        # Keep non-person-related detections
                        filtered_boxes.append(box)
                        filtered_class_ids.append(class_id)
                        filtered_scores.append(score)
                    else:
                        # Check for overlap with person boxes
                        grounded = False
                        for person_box in person_boxes:
                            iou = calculate_iou(box, person_box)
                            if iou > iou_threshold:
                                grounded = True
                                break

                        if grounded:
                            filtered_boxes.append(box)
                            filtered_class_ids.append(class_id)
                            filtered_scores.append(score)

                # Print inferred classes (using filtered results)
                inferred_classes = [names[class_id] for class_id in filtered_class_ids]
                print(f"Frame {frame_count}: Inferred classes - {inferred_classes}")

                # Check for violation (using filtered results)
                violation_classes = [1, 3, 5, 6, 9, 10]
                violation_list = []
                violation_class_ids = []  
                violation_boxes = [] 

                for i, class_id in enumerate(filtered_class_ids):
                    if class_id in violation_classes:
                        violation_list.append(names[class_id])
                        violation_class_ids.append(class_id)
                        violation_boxes.append(filtered_boxes[i])
                        
                # Draw detections on the frame (using filtered results)
                frame = draw_boxes(frame.copy(), violation_boxes, violation_class_ids, names)
                # combined_img = draw_boxes(frame.copy(), filtered_boxes, filtered_class_ids, names)
                
                if time.time() - last_datasend_time >= config['datasend_interval']:
                    start_time = time_to_string(last_datasend_time)
                    end_time = time_to_string(current_time)
                    
                    # Prepare data for sending
                    data = {
                        "sn": config['sn'],
                        "violation_list": json.dumps(violation_list),
                        "violation": True if len(violation_list) != 0 else False,
                        "start_time": start_time,
                        "end_time": end_time
                    }

                    # Prepare files for sending
                    files = {"image": mat_to_response(frame)}

                    # Send data with image
                    data_uploader.send_data(data, files=files)
                    last_datasend_time = time.time()
                
                # Send live stream data
                if config['livestream']:
                    streamer.updateFrame(frame)
                

                if config["show"]:
                    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
                    cv2.imshow("Output", frame)
                    key = cv2.waitKey(1)
                    if key == 27:
                        break
                    
                last_inference_time = current_time
    finally:
        cap_async.release()
        cv2.destroyAllWindows()        
        if config['livestream']:
            streamer.stop_streaming()

# Usage
if __name__ == '__main__':
    # Load configuration from config.json
    with open("config.json", "r") as f:
        config = json.load(f)

    # Set YOLOv8 to quiet mode
    # os.environ['YOLO_VERBOSE'] = 'False'
        
    # label mapping
    labels = ['hat','no_hat', 'mask','no_mask','gloves','no_gloves','food_uncover','pilgrim','no_pilgrim','waste','incorrect_mask','food_processing']
    names = {}
    for i, label in enumerate(labels):
        names[i] = label
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kitchen device: {device}")
        
    # Load the ONNX model
    model_path = config["model"]
    model = RKNN_instance(model_path, conf_thres=0.2, iou_thres=0.2, classes=(*labels,))
        
    # Load person detection for detection grounding
    # person_model = YOLO(config['person_model']).to(device) ### use this model for person bounding boxes
    labels = ['person']
    names = {}
    for i, label in enumerate(labels):
        names[i] = label
    person_model = RKNN_instance(model_path, conf_thres=0.2, iou_thres=0.2, classes=(*labels,))

    if not config["live"]:
        demo(model, config, names, person_model)
    else:
        live(model, config, names, person_model)