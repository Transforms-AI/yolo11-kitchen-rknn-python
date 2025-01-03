import cv2
from libraries.rknn import RKNN_instance
from libraries.datasend import DataUploader
from libraries.utils import time_to_string
import json
import time
import os


def demo(model: RKNN_instance, config, names):
    """
    Performs object detection on a single image.

    Args:
        model: The YOLO_ONNX model.
        config: Configuration dictionary loaded from config.json.
        names: Dictionary mapping class IDs to class names.
    """
    image = cv2.imread(config["demo_picture"])

    # Detect objects
    start = time.perf_counter()
    boxes, class_ids, scores = model.detect(image)
    print(
        f"Full pipeline processing time: {(time.perf_counter() - start)*1000:.2f} ms")

    # Draw detections on the image
    while True:
        combined_img = model.draw_detections(
            image.copy(), boxes, class_ids, scores)

        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.imshow("Output", combined_img)
        key = cv2.waitKey(1)  # Wait for 1 millisecond
        if key == 27:  # Check for ESC key (ASCII 27)
            break
    cv2.destroyAllWindows()

def live(model: RKNN_instance, config, names):
    """
    Performs object detection on a live video stream from a camera,
    and sends data to the server when specific classes are detected.

    Args:
        model: The YOLO_ONNX model.
        config: Configuration dictionary loaded from config.json.
        names: Dictionary mapping class IDs to class names.
    """
    video_source = config["video_source"]
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    frame_count = 0
    last_inference_time = time.time()
    last_data_sent_time = time.time()  # Initialize last data sent time

    # Initialize DataUploader
    api_url = config['datadata_send_url']
    headers = {"X-Secret-Key": config["X-Secret-Key"]}
    data_uploader = DataUploader(api_url, headers)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        frame_count += 1
        current_time = time.time()

        # Perform inference every 'seconds_per_frame' seconds
        if current_time - last_inference_time >= config["seconds_per_frame"]:
            # Detect objects
            boxes, class_ids, scores = model.detect(frame)

            # Print inferred classes
            inferred_classes = [names[class_id] for class_id in class_ids]
            print(
                f"Frame {frame_count}: Inferred classes - {inferred_classes}")
            
            # Check if violation
            violation_classes = {0, 1, 2, 3}  # Classes that indicate a violation, person is not a violation currently, check later
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
            combined_img = model.draw_detections(
                        frame.copy(), boxes, class_ids, scores)

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

            last_data_sent_time = current_time  # Update last data sent time

            if config["show"]:
                cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
                cv2.imshow("Output", combined_img)
                key = cv2.waitKey(1)  # Wait for 1 millisecond
                if key == 27:  # Check for ESC key (ASCII 27)
                    break
    cap.release()
    cv2.destroyAllWindows()


# Usage
if __name__ == '__main__':
    # Load configuration from config.json
    with open("config.json", "r") as f:
        config = json.load(f)

    # Load the ONNX model
    model_path = config["model"]
    model = RKNN_instance(model_path, conf_thres=0.2, iou_thres=0.2, classes=(
        "no uniform", "food uncovered", "no gloves", "no mask", "person"))

    # label mapping
    names = {
        0: "no uniform",
        1: "food uncovered",
        2: "no gloves",
        3: "no mask",
        4: "person"
    }

    if not config["live"]:
        demo(model, config, names)
    else:
        live(model, config, names)
