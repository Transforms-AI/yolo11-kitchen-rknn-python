import time
import cv2

UPLOAD_IMAGE_MAX_WIDTH_DEFAULT = 1920
JPEG_DEFAULT_QUALITY = 65

def time_to_string(input):
    time_tuple = time.gmtime(input)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time_tuple)

def resize_frame(frame, max_width=UPLOAD_IMAGE_MAX_WIDTH_DEFAULT):
    """
    Resizes an OpenCV frame to a maximum width while maintaining aspect ratio.

    Args:
        frame (np.ndarray): The image frame to resize.
        max_width (int): The maximum desired width. If the frame's width is
                         already less than or equal to max_width, the original
                         frame is returned.

    Returns:
        np.ndarray: The resized frame.
    """
    height, width = frame.shape[:2]

    if width > max_width:
        # Calculate the ratio
        ratio = max_width / width
        # Calculate new dimensions
        new_width = max_width
        new_height = int(height * ratio)
        # Resize the image using INTER_AREA for shrinking
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_frame
    else:
        # No resizing needed
        return frame

def mat_to_response(frame, max_width=UPLOAD_IMAGE_MAX_WIDTH_DEFAULT, jpeg_quality=JPEG_DEFAULT_QUALITY, filename="image.jpg"):
    """
    Resizes (if necessary) and encodes an OpenCV frame (NumPy array)
    to JPEG bytes in memory with a specified quality.

    Args:
        frame (np.ndarray): The image frame to encode.
        max_width (int): Maximum width for resizing before encoding.
        jpeg_quality (int): JPEG compression quality (0-100).

    Returns:
        tuple | None: A tuple suitable for the 'files' parameter in requests
                      (filename, image_bytes, content_type), or None if encoding fails.
    """
    try:
        # 1. Resize the frame
        resized_frame = resize_frame(frame, max_width)

        # 2. Encode the resized image to JPEG format in memory
        # Use the specified quality parameter
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        ret, encoded_image = cv2.imencode(".jpg", resized_frame, encode_params)

        if not ret:
            print("Error: Could not encode image.")
            return None

        # Convert the encoded image NumPy array to bytes
        image_bytes = encoded_image.tobytes()

        # Prepare the tuple for sending (filename, content_bytes, content_type)
        return (filename, image_bytes, "image/jpeg")
    except Exception as e:
        print(f"Error during image resizing or encoding: {e}")
        return None