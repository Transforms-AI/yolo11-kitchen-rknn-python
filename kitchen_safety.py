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
import socket # For local_ip resolution

logger = logging.getLogger("kitchen_safety_sequential")


def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0]); yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2]); yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    if (box1Area + box2Area - interArea) == 0: return 0.0
    return interArea / float(box1Area + box2Area - interArea)


def process_single_stream_cycle(
    frame: any, # The frame already read from VideoCaptureAsync
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

    # Heartbeat is independent of frame processing success for this cycle
    if current_time - timers['last_heartbeat_time'] >= global_config["heartbeat_interval"]:
        local_ip_for_heartbeat = stream_state.get('local_ip_resolved', "N/A")
        data_uploader.send_heartbeat(sn, config["local_ip"], time_to_string(current_time), status_log="Heartbeat OK")
        timers['last_heartbeat_time'] = current_time
        logger.debug(f"[{sn}] Heartbeat sent.")

    # Default display frame is the input frame
    display_frame = frame.copy()
    violation_list_for_send = []
    violation_occurred_for_send = False

    if current_time - timers['last_inference_time'] >= global_config["inference_interval"]:
        logger.debug(f"[{sn}] Performing inference (Frame {stream_state['frame_count']}).")
        use_half = (device == 'cuda')
        person_model_conf = stream_config.get('person_model_conf', 0.2)
        person_model_iou = stream_config.get('person_model_iou', 0.2)
        main_model_conf = stream_config.get('main_model_conf', 0.3)
        main_model_iou = stream_config.get('main_model_iou', 0.2)

        person_results = person_model.predict(frame, conf=person_model_conf, iou=person_model_iou, device=device, half=use_half, verbose=False)
        person_boxes = []
        if person_results and person_results[0].boxes:
            person_boxes = [b.xyxy[0].cpu().numpy() for b in person_results[0].boxes if int(b.cls.cpu()) == 0]

        results = main_model.predict(frame, conf=main_model_conf, iou=main_model_iou, device=device, half=use_half, verbose=False)
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

        violation_cfg = [1, 3, 5, 6, 9, 10]
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
            frame = draw_boxes(frame.copy(), viol_boxes_draw, viol_cids_draw, class_names, filtered_scores) 
        
        timers['last_inference_time'] = current_time

        # Data Sending
        if current_time - timers['last_datasend_time'] >= global_config['datasend_interval']:
            if violation_occurred_for_send or global_config.get("send_data_even_if_no_violation", True):
                data_payload = {
                    "sn": sn, 
                    "violation_list": json.dumps(violation_list_for_send),
                    "violation": violation_occurred_for_send,
                    "start_time": time_to_string(timers['last_datasend_time']),
                    "end_time": time_to_string(current_time)
                }
                img_name, img_bytes, img_type = mat_to_response(display_frame, max_width=config['frame_width'], jpeg_quality=config['frame_jpeg_quality'])
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
            streamer.updateFrame(frame)

        if global_config["show"]:
            cv2.imshow(f"Output_{sn}", frame)
            # cv2.waitKey(1) will be in the main loop


def sequential_multi_stream_loop(global_config, main_model, class_names, person_model, device):
    stream_configs = global_config.get('streams', [])
    if not stream_configs:
        logger.error("No streams defined in configuration.")
        return

    stream_states = {}
    data_uploader = DataUploader(
        global_config['data_send_url'], global_config['heartbeat_url'],
        {"X-Secret-Key": global_config["X-Secret-Key"]}
    )

    for s_conf in stream_configs:
        sn = s_conf['sn']
        logger.info(f"[{sn}] Initializing...")
        
        video_source_uri = s_conf.get("local_video_source") if s_conf.get('local_video') else s_conf["video_source"]
        is_looping = s_conf.get('local_video', False)

        # Initialize VideoCaptureAsync; it handles opening internally.
        cap = VideoCaptureAsync(src=video_source_uri, loop=is_looping,
                                width=s_conf.get("frame_width", 640), 
                                height=s_conf.get("frame_height", 480))
        cap.start() 

        resolved_local_ip = global_config['local_ip']
        if resolved_local_ip == 'auto':
            try: resolved_local_ip = socket.gethostbyname(socket.gethostname())
            except socket.gaierror: resolved_local_ip = "127.0.0.1"; logger.warning(f"[{sn}] Auto IP failed.")

        streamer_obj = None
        if global_config['livestream']:
            streamer_obj = StreamPublisher(f"live_{sn}", host=resolved_local_ip)
            streamer_obj.start_streaming()

        stream_states[sn] = {
            'cap': cap, 'streamer': streamer_obj, 'config': s_conf, 'frame_count': 0,
            'timers': {
                'last_inference_time': time.time() - global_config["inference_interval"] * 2,
                'last_datasend_time': time.time() - global_config["datasend_interval"] * 2,
                'last_heartbeat_time': time.time() - global_config["heartbeat_interval"] * 2
            },
            'local_ip_resolved': resolved_local_ip,
            'last_successful_read': time.time() # For tracking if a stream is dead
        }

    # Main Sequential Loop
    try:
        while True:
            active_stream_found_this_cycle = False
            for sn, state in stream_states.items():
                cap_obj = state['cap']
                
                grabbed, frame = cap_obj.read(wait_for_frame=False, timeout=0.01) 

                if grabbed and frame is not None:
                    active_stream_found_this_cycle = True
                    state['last_successful_read'] = time.time()
                    state['frame_count'] += 1
                    
                    process_single_stream_cycle(
                        frame, state, state['config'], global_config,
                        main_model, class_names, person_model, device, data_uploader
                    )
                    
                elif cap_obj.started: # Still started but no frame
                    # Log if it's been too long since a successful read.
                    if time.time() - state['last_successful_read'] > global_config.get("stream_read_timeout", 30): 
                        logger.warning(f"[{sn}] No frame read for over {global_config.get('stream_read_timeout', 30)}s. Stream might be down.")
                        
                else: # Not grabbed and not started (or thread died)
                    logger.error(f"[{sn}] Capture thread not running or failed to grab frame. Attempting to restart VideoCaptureAsync.")
                    state['cap'].release() 
                    video_source_uri = state['config'].get("local_video_source") if state['config'].get('local_video') else state['config']["video_source"]
                    is_looping = state['config'].get('local_video', False)
                    new_cap = VideoCaptureAsync(src=video_source_uri, loop=is_looping,
                                                width=state['config'].get("frame_width", 640),
                                                height=state['config'].get("frame_height", 480))
                    new_cap.start()
                    state['cap'] = new_cap
                    state['last_successful_read'] = time.time() # Reset timer
                    logger.info(f"[{sn}] Re-initialized and started VideoCaptureAsync.")


            if not active_stream_found_this_cycle and not stream_states:
                 logger.info("No streams configured or all failed critically. Exiting.")
                 break
            if not active_stream_found_this_cycle and stream_states:
                # All streams failed to provide a frame in this cycle.
                logger.debug("No active frames from any stream in this cycle. Pausing briefly.")
                time.sleep(0.1)


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
            if global_config["show"]: cv2.destroyWindow(f"Output_{sn}")
        if global_config["show"]: cv2.destroyAllWindows()
        logger.info("Application cleanup finished.")


if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("Usage: python guard_detection.py <config_path>...")
        sys.exit(1)

    paths = sys.argv[1:]
    print(f"Received paths: {paths}")

    config_path = paths[0]
    # detector = GeneralSystem(config_path)
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except Exception as e:
        logging.basicConfig(level=logging.ERROR) # Basic for this error
        logger = logging.getLogger("setup_error")
        logger.critical(f"Failed to load/parse config.json: {e}")
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

    labels = ['hat','no_hats', 'mask','no_masks','gloves','no_gloves','food_uncovered','pilgrim','no_pilgrim','garbage','incorrect_mask','food_processing']
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

    # Add a stream_read_timeout to global config if not present, for demonstration
    if 'stream_read_timeout' not in config:
        config['stream_read_timeout'] = 30 # seconds

    sequential_multi_stream_loop(config, main_model, class_names_map, person_model, device)