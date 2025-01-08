import cv2
from libraries.datasend import DataUploader
from libraries.utils import time_to_string
import json
import time
import os

def demo(model: 'RKNN_instance | YOLO', config, names):
    """
    Performs object detection on a video stream using either RKNN or YOLO (Ultralytics).

    Args:
        model: The RKNN model instance or YOLO model.
        config: Configuration dictionary loaded from config.json.
        names: Dictionary mapping class IDs to class names (used for RKNN).
    """

    video_source = config["local_video_source"]

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return

    cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)

    is_yolo = isinstance(model, YOLO)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video stream or error reading frame.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Detect objects
        start = time.perf_counter()

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
            names = result.names

        else:
            # RKNN inference
            boxes, class_ids, scores = model.detect(frame)
            inference_time = (time.perf_counter() - start) * 1000
            print(f"Inference time (RKNN): {inference_time:.2f} ms")

        # Draw detections on the frame
        if is_yolo:
            combined_img = draw_yolo_detections(frame.copy(), boxes, class_ids, scores, names)
        else:
            combined_img = model.draw_detections(frame.copy(), boxes, class_ids, scores, names)

        cv2.imshow("Output", combined_img)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def draw_yolo_detections(image, boxes, class_ids, scores, names):
    """
    Draws bounding boxes and labels on the image for YOLO results.

    Args:
        image: The input image (NumPy array).
        boxes: Bounding boxes (NumPy array [N, 4] in xyxy format).
        class_ids: Class IDs (NumPy array [N]).
        scores: Confidence scores (NumPy array [N]).
        names: Dictionary mapping class IDs to names.
    """
    for box, class_id, score in zip(boxes, class_ids, scores):
        x1, y1, x2, y2 = map(int, box)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display class name and score
        label = f"{names[class_id]}: {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

def live(model: 'RKNN_instance | YOLO', config, names):
    """
    Performs object detection on a live video stream from a camera,
    and sends data to the server when specific classes are detected.

    Args:
        model: The RKNN model instance or YOLO model.
        config: Configuration dictionary loaded from config.json.
        names: Dictionary mapping class IDs to class names (used for RKNN).
    """
    video_source = config["video_source"]
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    frame_count = 0
    last_inference_time = time.time()
    last_data_sent_time = time.time()

    # Initialize DataUploader
    api_url = config['datadata_send_url']
    headers = {"X-Secret-Key": config["X-Secret-Key"]}
    data_uploader = DataUploader(api_url, headers)

    is_yolo = isinstance(model, YOLO)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        frame_count += 1
        current_time = time.time()

        # Perform inference every 'seconds_per_frame' seconds
        if current_time - last_inference_time >= config["seconds_per_frame"]:
            last_inference_time = current_time  # Update last inference time

            if is_yolo:
                # YOLO (Ultralytics) inference
                results = model(frame)
                result = results[0]
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                scores = result.boxes.conf.cpu().numpy()
                names = result.names  # Get names from YOLO results

            else:
                # RKNN inference
                boxes, class_ids, scores = model.detect(frame)

            # Print inferred classes
            inferred_classes = [names[class_id] for class_id in class_ids]
            print(f"Frame {frame_count}: Inferred classes - {inferred_classes}")

            # Check for violation
            violation_classes = {1, 2, 3}
            has_violation = any(class_id in violation_classes for class_id in class_ids)
            violation_list = []
            if has_violation:
                violation_list = [
                    names[class_id]
                    for class_id in class_ids
                    if class_id in violation_classes
                ]

            start_time = time_to_string(last_data_sent_time)
            end_time = time_to_string(current_time)

            # Prepare data for sending
            data = {
                "sn": config['sn'],
                "violation": has_violation,
                "violation_list": violation_list,
                "start_time": start_time,
                "end_time": end_time
            }

            # Draw detections on the frame
            if is_yolo:
                combined_img = draw_yolo_detections(frame.copy(), boxes, class_ids, scores, names)
            else:
                combined_img = model.draw_detections(frame.copy(), boxes, class_ids, scores, names)

            # Save the image temporarily
            temp_image_path = "temp_image.jpg"
            cv2.imwrite(temp_image_path, combined_img)

            # Prepare files for sending
            files = {"image": open(temp_image_path, "rb")}

            # Send data with image
            messages = data_uploader.send_data(data, files=files)

            # Remove temp image
            os.remove(temp_image_path)

            # Print messages from data sending
            for msg in messages:
                print(msg)

            last_data_sent_time = current_time

            if config["show"]:
                cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
                cv2.imshow("Output", combined_img)
                key = cv2.waitKey(1)
                if key == 27:
                    break

    cap.release()
    cv2.destroyAllWindows()


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
        from ultralytics import YOLO
        model = YOLO(model_path)

    if not config["live"]:
        demo(model, config, names)
    else:
        live(model, config, names)
