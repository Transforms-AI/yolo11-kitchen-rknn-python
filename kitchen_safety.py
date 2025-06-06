import sys
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
import logging

# --- Script Information ---
__version__ = "2.31"
__author__ = "TransformsAI"

logger = logging.getLogger("kitchen_safety")


def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0]); yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2]); yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    if (box1Area + box2Area - interArea) == 0: return 0.0
    return interArea / float(box1Area + box2Area - interArea)


def process_single_stream_cycle(
    frame: any, 
    stream_state: dict,
    stream_config: dict,
    global_config: dict,
    main_model: YOLO,
    class_names: dict,
    person_model: YOLO,
    device: str,
    data_uploader: DataUploader
):
    """
    Processes one cycle for a single stream given an already acquired frame.
    Modifies stream_state in place (timers, etc.).
    """
    sn = stream_config['sn']
    streamer = stream_state.get('streamer')
    timers = stream_state['timers']
    current_time = time.time()

    # Keep original frame for server upload
    original_frame = frame.copy()
    
    status_text = f"Processing Frame: {stream_state['frame_count']}, Time: {time_to_string(current_time)}"

    # Heartbeat is independent of frame processing success for this cycle
    if current_time - timers['last_heartbeat_time'] >= global_config["heartbeat_interval"]:
        data_uploader.send_heartbeat(sn, time_to_string(current_time), status_log=status_text)
        timers['last_heartbeat_time'] = current_time

    # Default display frame is the resized frame
    display_frame = original_frame.copy()
    violation_list_for_send = []
    violation_occurred_for_send = False

    if current_time - timers['last_inference_time'] >= global_config["inference_interval"]:
        logger.debug(f"[{sn}] Performing inference (Frame {stream_state['frame_count']}).")
        use_half = (device == 'cuda')
        person_model_conf = stream_config.get('person_model_conf', 0.2)
        person_model_iou = stream_config.get('person_model_iou', 0.2)
        main_model_conf = stream_config.get('main_model_conf', 0.3)
        main_model_iou = stream_config.get('main_model_iou', 0.2)

        # Use inference_frame for model predictions
        person_results = person_model.predict(original_frame, imgsz = stream_config.get("frame_width", 640), conf=person_model_conf, iou=person_model_iou, device=device, half=use_half, verbose=True)
        person_boxes = []
        if person_results and person_results[0].boxes:
            person_boxes = [b.xyxy[0].cpu().numpy() for b in person_results[0].boxes if int(b.cls.cpu()) == 0]

        results = main_model.predict(original_frame, imgsz = stream_config.get("frame_width", 640), conf=main_model_conf, iou=main_model_iou, device=device, half=use_half, verbose=True)
        boxes, class_ids, scores = [], [], []
        if results and results[0].boxes:
            boxes = [b.xyxy[0].cpu().numpy() for b in results[0].boxes]
            class_ids = [int(b.cls.cpu()) for b in results[0].boxes]
            scores = [float(b.conf.cpu()) for b in results[0].boxes]

        filtered_boxes, filtered_class_ids, filtered_scores = [], [], []
        person_related_classes = [0, 1, 2, 3, 4, 5, 8, 10]
        iou_thr = global_config['iou_threshold']
        for i, box_coords in enumerate(boxes):
            cid = class_ids[i]
            if cid not in person_related_classes or any(calculate_iou(box_coords, p_box) > iou_thr for p_box in person_boxes):
                filtered_boxes.append(box_coords)
                filtered_class_ids.append(cid)
                filtered_scores.append(scores[i])

        inferred_names = [class_names[cid] for cid in filtered_class_ids if cid in class_names]
        logger.info(f"[{sn}] Inferred: {inferred_names}")

        # violation_cfg = [1, 3, 5, 6, 9, 10]
        violation_cfg = [1, 3, 5, 6, 7, 9]
        current_violations, viol_boxes_draw, viol_cids_draw = [], [], []
        for i, cid in enumerate(filtered_class_ids):
            if cid in violation_cfg and cid in class_names:
                current_violations.append(class_names[cid])
                viol_boxes_draw.append(filtered_boxes[i])
                viol_cids_draw.append(cid)
        
        violation_list_for_send = list(set(current_violations))
        violation_occurred_for_send = bool(violation_list_for_send) 
        logger.debug(f"[{sn}] Violations: {violation_list_for_send}")

        if global_config['draw']:
            display_frame = draw_boxes(display_frame, viol_boxes_draw, viol_cids_draw, class_names, filtered_scores) 
        
        timers['last_inference_time'] = current_time

        # Data Sending - use original frame for server upload
        if current_time - timers['last_datasend_time'] >= global_config['datasend_interval']:
            if violation_occurred_for_send or global_config.get("send_data_even_if_no_violation", True):
                data_payload = {
                    "sn": sn, 
                    "violation_list": json.dumps(violation_list_for_send),
                    "violation": violation_occurred_for_send,
                    "start_time": time_to_string(timers['last_datasend_time']),
                    "end_time": time_to_string(current_time)
                }
                # Use separate dimensions and quality for server upload
                send_width = global_config.get("frame_send_width", global_config.get("frame_width", 640))
                send_quality = global_config.get("frame_send_jpeg_quality", global_config.get("frame_jpeg_quality", 85))
                
                # Use original_frame for server upload with send-specific settings
                img_name, img_bytes, img_type = mat_to_response(original_frame, max_width=send_width, jpeg_quality=send_quality, timestamp=current_time)
                files = {"image": (img_name, img_bytes, img_type)}

                if global_config.get('send_data', True):
                    data_uploader.send_data(data_payload, files=files)
                else:
                    logger.debug(f"[{sn}] Data sending skipped (send_data is False).")
                timers['last_datasend_time'] = current_time
            elif not violation_occurred_for_send:
                logger.debug(f"[{sn}] No new violations, data send skipped for this interval.")
                timers['last_datasend_time'] = current_time 

        if global_config['livestream'] and streamer:
            # Use display_frame (resized) for livestream
            streamer.updateFrame(display_frame)

        if global_config["show"]:
            # Use display_frame (resized) for local display
            cv2.imshow(f"Output_{sn}", display_frame)
            # cv2.waitKey(1) will be in the main loop


def sequential_multi_stream_loop(global_config, main_model, class_names, person_model, device):
    stream_configs = global_config.get('streams', [])
    if not stream_configs:
        logger.error("No streams defined in configuration.")
        return

    stream_states = {}
    data_uploader = DataUploader(
        global_config['data_send_url'], 
        global_config['heartbeat_url'],
        {"X-Secret-Key": global_config["X-Secret-Key"]},
        max_retries=config.get("max_send_retries", 5),
        retry_delay=config.get("send_retry_delay", 5),
        timeout=config.get("send_timeout", 30),
        debug=True,
        project_version= __version__,
    )

    for s_conf in stream_configs:
        sn = s_conf['sn']
        logger.info(f"[{sn}] Initializing...")
        
        video_source_uri = s_conf.get("local_video_source") if s_conf.get('local_video') else s_conf["video_source"]
        is_looping = s_conf.get('local_video', False)

        # Initialize VideoCaptureAsync; it handles opening internally.
        heartbeat_config = {
            'enabled': True,
            'sn': sn,
            'heartbeat_url': global_config.get('heartbeat_url'),
            'headers': {"X-Secret-Key": global_config.get("X-Secret-Key", "")},
            'interval': global_config.get("heartbeat_interval", 30)
        }
        
        cap = VideoCaptureAsync(src=video_source_uri, 
                                loop=is_looping,
                                heartbeat_config=heartbeat_config,
                                auto_restart_on_fail=True)
        cap.start() 

        streamer_obj = None
        if global_config['livestream']:
            streamer_obj = StreamPublisher(
                f"live_{sn}",
                host=global_config.get('local_ip', '127.0.0.1'),
                jpeg_quality=70, 
                target_width=1600
            )
            streamer_obj.start_streaming()

        stream_states[sn] = {
            'cap': cap, 'streamer': streamer_obj, 'config': s_conf, 'frame_count': 0,
            'timers': {
                'last_inference_time': time.time() - global_config["inference_interval"] * 2,
                'last_datasend_time': time.time() - global_config["datasend_interval"] * 2,
                'last_heartbeat_time': time.time() - global_config["heartbeat_interval"] * 2,
            }
        }

    # Main Sequential Loop
    try:
        while True:            
            
            for sn, state in stream_states.items():
                cap_obj = state['cap']
                try:
                    grabbed, frame = cap_obj.read(wait_for_frame=False, timeout=0.01)
                    
                    if grabbed and frame is not None:
                        state['frame_count'] += 1
                        
                        process_single_stream_cycle(
                            frame, state, state['config'], global_config,
                            main_model, class_names, person_model, device, data_uploader
                        )

                except Exception as e:
                    error_msg = f"Exception during frame processing: {str(e)}"
                    data_uploader.send_heartbeat(
                        config['sn'],
                        time_to_string(time_to_string(time.time())),
                        status_log=error_msg
                    )

            if global_config["show"]:
                key = cv2.waitKey(1) & 0xFF
                if key == 27: # ESC
                    logger.info("ESC key pressed, initiating shutdown...")
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        logger.info("Shutdown initiated.")
    except Exception as e:
        logger.critical(f"Unhandled exception in main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up resources...")
        for sn, state in stream_states.items():
            if state.get('cap'): state['cap'].release()
            if state.get('streamer'): state['streamer'].stop_streaming()
        if global_config["show"]: cv2.destroyAllWindows()
        data_uploader.shutdown()
        logger.info("Application cleanup finished.")


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: python kitchen_safety.py <config_path>...")
        sys.exit(1)

    paths = sys.argv[1:]
    print(f"Received paths: {paths}")

    config_path = paths[0]
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        logging.basicConfig(level=logging.ERROR)
        logger = logging.getLogger("setup_error")
        logger.critical(f"Failed to load/parse {config_path}: {e}")
        exit(1)
    
    log_level_str = config.get('logging_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Configure main logger
    logger.setLevel(log_level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.info(f"Logging level set to {log_level_str}.")

    # violation_cfg = [1, 3, 5, 6, 7, 9, 10]
    # labels = ['hat','no_hats', 'mask','no_masks','gloves','no_gloves','food_uncovered','pilgrim','no_pilgrim','garbage','incorrect_mask','food_processing']
    labels = ['hat','no_hats', 'mask','no_masks','gloves','no_gloves','food_uncovered','uniform_missing','no_pilgrim','garbage','no_masks','food_processing'] # incorrect mask being used as no_mask
    class_names_map = {i: label for i, label in enumerate(labels)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    try:
        main_model = YOLO(config["model"], task='detect')
        person_model = YOLO(config['person_model'], task='detect')
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load YOLO models: {e}", exc_info=True)
        exit(1)

    sequential_multi_stream_loop(config, main_model, class_names_map, person_model, device)
