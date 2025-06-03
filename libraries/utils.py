import time
import cv2
import datetime
import numpy as np

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
        # 0. Replate timestamp
        frame = hide_camera_timestamp_and_add_current_time(frame)
        
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

def hide_camera_timestamp_and_add_current_time(
    frame,
    camera_ts_rect_coords=None,
    camera_ts_rect_ratios=(0.015, 0.05, 0.25, 0.035), 
    hide_rect_color=(255, 255, 255),
    new_ts_position_on_rect=True,
    new_ts_custom_position=None,
    new_ts_font=cv2.FONT_HERSHEY_SIMPLEX,
    new_ts_font_scale=None,
    new_ts_font_color=(0, 0, 0),
    new_ts_font_thickness=1,
    new_ts_padding_ratio=0.1 
):
    """
    Hides a region on a frame (defined by pixel coordinates or ratios)
    with a rectangle and adds the current system time (centered) onto that rectangle
    or at a custom position.

    Args:
        frame (np.ndarray): The input video frame (OpenCV BGR format).
        camera_ts_rect_coords (tuple, optional): (x, y, w, h) in pixels. Overrides ratios.
        camera_ts_rect_ratios (tuple, optional): (x_r, y_r, w_r, h_r) ratios (0.0-1.0).
                                       Default: (0.015, 0.05, 0.26, 0.035).
        hide_rect_color (tuple, optional): BGR color of hiding rectangle. Default: white.
        new_ts_position_on_rect (bool, optional): True to place new TS on hiding rect. Default: True.
        new_ts_custom_position (tuple, optional): (x,y) for new TS if not on rect. Default: None.
        new_ts_font (int, optional): Font type. Default: cv2.FONT_HERSHEY_SIMPLEX.
        new_ts_font_scale (float, optional): Font scale. If None, auto-calculated. Default: None.
        new_ts_font_color (tuple, optional): BGR color for new TS. Default: black.
        new_ts_font_thickness (int, optional): Thickness for new TS. Default: 1.
        new_ts_padding_ratio (float, optional): Padding for new TS within hiding rect,
                                                as a ratio of the rectangle's smaller dimension.
                                                Default: 0.1.

    Returns:
        np.ndarray: The modified frame.
    """
    output_frame = frame.copy()
    frame_h, frame_w = output_frame.shape[:2]

    # 1. Determine the rectangle coordinates (pixels)
    if camera_ts_rect_coords:
        rect_x, rect_y, rect_w, rect_h = camera_ts_rect_coords
    elif camera_ts_rect_ratios:
        xr, yr, wr, hr = camera_ts_rect_ratios
        rect_x = int(frame_w * xr)
        rect_y = int(frame_h * yr)
        rect_w = int(frame_w * wr)
        rect_h = int(frame_h * hr)
        rect_w = max(1, rect_w)
        rect_h = max(1, rect_h)
    else:
        print("Warning: No rectangle coordinates or ratios. Using small default.")
        rect_x, rect_y, rect_w, rect_h = 10, 10, 100, 20

    # 2. Hide the area with a filled rectangle
    cv2.rectangle(output_frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h),
                  hide_rect_color, -1)

    # 3. Get current time and format it
    current_unix_time = time.time()
    dt_object = datetime.datetime.fromtimestamp(current_unix_time)
    time_string = dt_object.strftime("%Y-%m-%d %H:%M:%S")

    # 4. Determine font scale for the new timestamp
    if new_ts_font_scale is None:
        calculated_scale = rect_h * 0.022 # Factor for font scale based on rect height
        final_new_ts_font_scale = np.clip(calculated_scale, 0.3, 2.5) # Min 0.3, Max 2.5
    else:
        final_new_ts_font_scale = new_ts_font_scale
    
    final_new_ts_font_thickness = max(1, int(final_new_ts_font_scale + 0.5)) if new_ts_font_thickness == 1 else new_ts_font_thickness

    # 5. Determine position for the new timestamp
    if new_ts_position_on_rect:
        # Calculate padding in pixels. Using smaller dimension of rect for reference.
        # This makes padding more consistent if rect is very wide or very tall.
        padding_ref_dim = min(rect_w, rect_h)
        padding_pixels = int(padding_ref_dim * new_ts_padding_ratio)

        (text_w_px, text_h_above_baseline_px), baseline_px = cv2.getTextSize(
            time_string, new_ts_font, final_new_ts_font_scale, final_new_ts_font_thickness
        )
        full_text_render_height_px = text_h_above_baseline_px + baseline_px

        # Horizontal centering
        available_w_for_text = rect_w - (2 * padding_pixels)
        if text_w_px > available_w_for_text: # Text wider than available space
            ts_x_pos = rect_x + padding_pixels # Align to left padding
        else:
            horizontal_offset = (available_w_for_text - text_w_px) // 2
            ts_x_pos = rect_x + padding_pixels + horizontal_offset

        # Vertical centering
        available_h_for_text = rect_h - (2 * padding_pixels)
        if full_text_render_height_px > available_h_for_text: # Text taller than available space
            # Align top of text (baseline - text_h_above_baseline) with top padding
            ts_y_pos = rect_y + padding_pixels + text_h_above_baseline_px
        else:
            # Text fits, center it vertically
            vertical_offset = (available_h_for_text - full_text_render_height_px) // 2
            ts_y_pos = rect_y + padding_pixels + vertical_offset + text_h_above_baseline_px
        
        final_ts_position = (ts_x_pos, ts_y_pos)

    elif new_ts_custom_position:
        final_ts_position = new_ts_custom_position
    else:
        final_ts_position = (10, frame_h - 10) # Fallback

    # 6. Put the new timestamp on the frame
    cv2.putText(output_frame,
                time_string,
                final_ts_position,
                new_ts_font,
                final_new_ts_font_scale,
                new_ts_font_color,
                final_new_ts_font_thickness,
                cv2.LINE_AA)

    return output_frame
