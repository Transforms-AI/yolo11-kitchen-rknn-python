import time
import cv2

def time_to_string(input):
    time_tuple = time.gmtime(input)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time_tuple)

def mat_to_response(frame):
    # Encode the image to JPEG format in memory
    ret, encoded_image = cv2.imencode(".jpg", frame)
    if not ret:
        print("Error: Could not encode image.")
        return

    # Convert the encoded image to bytes
    image_bytes = encoded_image.tobytes()

    # Prepare files for sending
    return ("image.jpg", image_bytes, "image/jpeg")