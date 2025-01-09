import cv2
import numpy as np

def draw_boxes(frame, boxes, classes, class_to_label, confidences=None):
    """
    Draws bounding boxes with labels and optional confidences on an image.

    Args:
        frame: The image (as a NumPy array) on which to draw.
        boxes: A list of bounding boxes, each represented as [x_min, y_min, x_max, y_max].
        classes: A list of class indices corresponding to each box.
        class_to_label: A dictionary mapping class indices to label names.
        confidences: (Optional) A list of confidence scores corresponding to each box.
    """

    frame_height, frame_width = frame.shape[:2]

    # Calculate text size relative to image width
    text_size = max(1, int(frame_width / 500))  # Adjust 500 for desired scaling
    text_thickness = max(1, int(frame_width / 500))

    # Generate a color palette for unique classes
    unique_classes = sorted(list(set(classes)))
    color_palette = generate_color_palette(len(unique_classes))
    class_to_color = {cls: color_palette[i] for i, cls in enumerate(unique_classes)}

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = [int(coord) for coord in box]
        class_index = classes[i]
        label = class_to_label[class_index]
        color = class_to_color[class_index]

        # Add confidence if available
        if confidences is not None:
            confidence = confidences[i]
            label = f"{label}: {confidence:.2f}"

        # Draw the bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # Calculate text background size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness
        )

        # Draw text background
        cv2.rectangle(
            frame,
            (x_min, y_min - text_height - baseline),
            (x_min + text_width, y_min),
            color,
            -1,
        )

        # Draw the label text
        cv2.putText(
            frame,
            label,
            (x_min, y_min - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            (0, 0, 0),  # Black text
            text_thickness,
            cv2.LINE_AA,
        )

    return frame

def generate_color_palette(num_colors):
    """
    Generates a list of visually distinct colors.

    Args:
        num_colors: The number of colors to generate.

    Returns:
        A list of (B, G, R) tuples representing the colors.
    """

    if num_colors == 0:
        return []  # Return an empty list if no colors are requested

    # Use HSV color space for better distinct color generation
    hsv_colors = np.array(
        [[i / num_colors, 1, 1] for i in range(num_colors)], dtype=np.float32
    )
    rgb_colors = cv2.cvtColor(np.array([hsv_colors]), cv2.COLOR_HSV2BGR)[0]

    # Convert to (B, G, R) tuples and scale to 0-255
    color_palette = [(int(b * 255), int(g * 255), int(r * 255)) for b, g, r in rgb_colors]

    return color_palette