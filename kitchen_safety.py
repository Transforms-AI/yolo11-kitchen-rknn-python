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
from ultralytics import YOLO  # Import YOLO from ultralytics

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

    # Compute the area of both the prediction and ground-truth
    # rectangles
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou

def demo(model: 'RKNN_instance | YOLO', config, names, person_model):
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
    cap_async = VideoCaptureAsync(video_source)
    cap_async.start()

    cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)

    is_yolo = isinstance(model, YOLO)

    while True:
        ret, frame = cap_async.read()

        if not ret:
            print("End of video stream or error reading frame.")
            cap_async.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Detect objects
        start = time.perf_counter()

        # --- Person Detection ---
        person_results = person_model(frame)
        person_result = person_results[0]
        person_boxes = person_result.boxes.xyxy.cpu().numpy()
        person_class_ids = person_result.boxes.cls.cpu().numpy().astype(int)

        # Filter for person class (assuming class ID 0 represents 'person')
        person_boxes = [box for i, box in enumerate(person_boxes) if person_class_ids[i] == 0]

        if is_yolo:
            # YOLO (Ultralytics) inference
            results = model(frame)  # Assuming you want to display results on the original frame
            inference_time = (time.perf_counter() - start) * 1000
            print(f"Inference time (YOLO): {inference_time:.2f} ms")

            # Use results for drawing (see next step)
            result = results[0]  # Assuming you process one frame at a time
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()

        else:
            # RKNN inference
            boxes, class_ids, scores = model.detect(frame)
            inference_time = (time.perf_counter() - start) * 1000
            print(f"Inference time (RKNN): {inference_time:.2f} ms")

        # --- Grounding with Person Detections ---
        filtered_boxes = []
        filtered_class_ids = []
        filtered_scores = []

        # Check for violation (using filtered results)
        violation_classes = [1, 3, 5, 6, 9, 10]
        violation_list = []
        violation_class_ids = []  
        violation_boxes = [] 

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

        for i, class_id in enumerate(filtered_class_ids):
            if class_id in violation_classes:
                violation_list.append(names[class_id])
                violation_class_ids.append(class_id)
                violation_boxes.append(filtered_boxes[i])

        # Draw detections on the frame
        # combined_img = draw_boxes(frame.copy(), filtered_boxes, filtered_class_ids, names)
        combined_img = draw_boxes(frame.copy(), violation_boxes, violation_class_ids, names)

        cv2.imshow("Output", combined_img)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap_async.release()
    cv2.destroyAllWindows()

def live(model: 'RKNN_instance | YOLO', config, names, person_model):
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
    last_data_sent_time = time.time()
    last_heartbeat_time = time.time()

    # Initialize DataUploader
    api_url = config['data_send_url']
    heartbeat_url = config['heartbeat_url']
    headers = {"X-Secret-Key": config["X-Secret-Key"]}
    data_uploader = DataUploader(api_url, heartbeat_url, headers)
    

    is_yolo = isinstance(model, YOLO)
    
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
                messages = data_uploader.send_heartbeat(
                    config["sn"], config["local_ip"], time_to_string(current_time)
                )
                print(f"Heartbeat Response: {str(messages)}")
                last_heartbeat_time = current_time
            
            # Send live stream data
            if config['livestream']:
                streamer.updateFrame(frame)
                
            # Perform inference every 'seconds_per_frame' seconds
            if current_time - last_data_sent_time >= config["seconds_per_frame"]:
                
                # --- Person Detection ---
                person_results = person_model(frame)
                person_result = person_results[0]
                person_boxes = person_result.boxes.xyxy.cpu().numpy()
                person_class_ids = person_result.boxes.cls.cpu().numpy().astype(int)

                # Filter for person class (assuming class ID 0 represents 'person')
                person_boxes = [box for i, box in enumerate(person_boxes) if person_class_ids[i] == 0]

                if is_yolo:
                    # YOLO (Ultralytics) inference
                    results = model(frame)
                    result = results[0]
                    boxes = result.boxes.xyxy.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    scores = result.boxes.conf.cpu().numpy()

                else:
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

                start_time = time_to_string(last_data_sent_time)
                end_time = time_to_string(current_time)
                
                # Prepare data for sending
                data = {
                    "sn": config['sn'],
                    "violation_list": json.dumps(violation_list),
                    "violation": True if len(violation_list) != 0 else False,
                    "start_time": start_time,
                    "end_time": end_time
                }

                # Draw detections on the frame (using filtered results)
                # combined_img = draw_boxes(frame.copy(), violation_boxes, violation_class_ids, names)
                combined_img = draw_boxes(frame.copy(), filtered_boxes, filtered_class_ids, names)

                # Prepare files for sending
                files = {"image": mat_to_response(combined_img)}
                # files = None

                # Send data with image
                print(data)
                messages = data_uploader.send_data(data, files=files)

                # Print messages from data sending
                print(f"Kitchen Datasend Response: {str(messages)}")
                last_data_sent_time = current_time

                if config["show"]:
                    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
                    cv2.imshow("Output", combined_img)
                    key = cv2.waitKey(1)
                    if key == 27:
                        break
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
        
    # label mapping
    labels = ['hat','no_hat', 'mask','no_mask','gloves','no_gloves','food_uncover','pilgrim','no_pilgrim','waste','incorrect_mask','food_processing']
    names = {}
    for i, label in enumerate(labels):
        names[i] = label
        
    # Load the ONNX model
    model_path = config["model"]
    if ".rknn" in model_path:
        from libraries.rknn import RKNN_instance
        model = RKNN_instance(model_path, conf_thres=0.2, iou_thres=0.2, classes=(*labels,))
    else:
        model = YOLO(model_path)
        
    # Load person detection for detection grounding
    person_model = YOLO(config['person_model']) ### use this model for person bounding boxes

    if not config["live"]:
        demo(model, config, names, person_model)
    else:
        live(model, config, names, person_model)