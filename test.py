#Load config.json
import json
import logging
import torch
import sys
from ultralytics import YOLO

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load configuration
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        logger.info("Configuration loaded successfully.")
    except FileNotFoundError:
        logger.critical("Configuration file 'config.json' not found.", exc_info=True)
        exit(1)
    except json.JSONDecodeError:
        logger.critical("Error decoding JSON from 'config.json'.", exc_info=True)
        exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # if config["model"] is not set, then try config["yoloe"]
    if config.get("yolo"):
        model_path = config["yolo"]
        logger.info(f"Using YOLO model from: {model_path}")
    elif config.get("detection_model"):
        model_path = config["detection_model"]
        logger.info(f"Using detection model from: {model_path}")
    else:
        model_path = config["model"]
        logger.info(f"Using model from: {model_path}")

    try:
        model = YOLO(model_path, task='detect')
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load YOLO models: {e}", exc_info=True)
        exit(1)

    results = model.predict(source="demo/demo-1.jpg", conf=0.25, save=True, save_txt=True, save_conf=True, device=device)

    # if model successfully inferred then return 0
    if results:
        print("Model inference successful.")
    else:   
        print("Model inference failed.")
        exit(1)

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)