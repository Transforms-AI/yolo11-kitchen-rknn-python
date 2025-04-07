import cv2
import numpy as np
import os
from .rknn_executor import RKNN_model_container

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)  # (width, height)
CLASSES = ("person", "bicycle")

class RKNN_instance:
    def __init__(self, model_path, target='rk3588', device_id=None, conf_thres = OBJ_THRESH, iou_thres = NMS_THRESH, classes = CLASSES, img_size = IMG_SIZE, model_version = 'v11', anchors = 'anchors.txt'):
        self.model_path = model_path
        self.target = target
        self.device_id = device_id
        self.model = None
        self.img_size = img_size
        self.classes = classes
        self.obj_thresh = conf_thres
        self.nms_thresh = iou_thres
        self.model_version = model_version
        self.anchor_path = anchors
        
        self._load_model()

    def _load_model(self):
        if not self.model_path.endswith('.rknn'):
            raise ValueError("Only .rknn model is supported in this version.")
        if self.model_version == 'v5':
            self._load_anchors()
        self.model = RKNN_model_container(self.model_path, self.target, self.device_id)
        
    def _load_anchors(self):
        with open(self.anchor_path, 'r') as f:
            values = [float(_v) for _v in f.readlines()]
            self.anchors = np.array(values).reshape(3,-1,2).tolist()
        print("use anchors from '{}', which is {}".format(self.anchor_path, self.anchors))

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with object threshold."""
        box_confidences = box_confidences.reshape(-1)
        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        _class_pos = np.where(class_max_score * box_confidences >= self.obj_thresh)
        scores = (class_max_score * box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores

    def _nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes."""
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    def _dfl(self, position):
        """Distribution Focal Loss (DFL)"""
        import torch
        x = torch.tensor(position)
        n, c, h, w = x.shape
        p_num = 4
        mc = c // p_num
        y = x.reshape(n, p_num, mc, h, w)
        y = y.softmax(2)
        acc_metrix = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)
        y = (y * acc_metrix).sum(2)
        return y.numpy()

    def _box_process(self, position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.img_size[1] // grid_h, self.img_size[0] // grid_w]).reshape(1, 2, 1, 1)

        position = self._dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

        return xyxy
   
    def _post_process(self, input_data):
        """
        Performs post-processing on the raw output of a YOLO model.

        Args:
            input_data (list): A list of numpy arrays, representing the output
                            from the model's detection heads.

        Returns:
            tuple: A tuple containing (boxes, classes, scores) for the final
                detections, or (None, None, None) if no detections are found.
                - boxes: numpy array of shape (N, 4) [x1, y1, x2, y2]
                - classes: numpy array of shape (N,) [class_id]
                - scores: numpy array of shape (N,) [confidence_score]
        """
        boxes, scores, classes_conf = [], [], []

        # --- Version-specific initial processing ---
        if self.model_version == 'v5':
            reshaped_input = []
            for i, _in in enumerate(input_data):
                # Remove batch dimension if present (assuming batch size 1)
                if _in.shape[0] == 1:
                    _in = _in.squeeze(0) # Shape becomes C*h*w, e.g., 255*80*80
                
                anchors_per_loc = len(self.anchors[i]) # e.g., 3
                # num_classes = 80 # Example
                # channels_per_anchor = 5 + num_classes # 4 box + 1 obj + num_classes
                channels_per_anchor = _in.shape[0] // anchors_per_loc # e.g. 255 // 3 = 85
                
                # Reshape: C*h*w -> anchors_per_loc * channels_per_anchor * h * w
                current_head_reshaped = _in.reshape(anchors_per_loc, channels_per_anchor, *_in.shape[-2:])
                reshaped_input.append(current_head_reshaped)

            # Extract boxes, scores (objectness), and class confidences
            for i in range(len(reshaped_input)):
                current_anchors = self.anchors[i]
                # Box processing needs anchors for v5
                boxes.append(self._box_process_v5(reshaped_input[i][:, :4, :, :], current_anchors))
                # Objectness score
                scores.append(reshaped_input[i][:, 4:5, :, :])
                # Class confidences
                classes_conf.append(reshaped_input[i][:, 5:, :, :])

        else:
            # Expect input_data structured differently, e.g., pairs per branch
            default_branch = 3
            if len(input_data) % default_branch != 0:
                raise ValueError(f"Input data length ({len(input_data)}) not compatible with default_branch ({default_branch}) for v11.")
            pair_per_branch = len(input_data) // default_branch # Number of tensors per branch

            for i in range(default_branch):
                # Box data index
                box_data_index = pair_per_branch * i
                # Remove batch dim if present
                box_tensor = input_data[box_data_index]
                if box_tensor.shape[0] == 1:
                    box_tensor = box_tensor.squeeze(0)
                boxes.append(self._box_process_v11(box_tensor))

                # Class confidence index
                class_conf_index = pair_per_branch * i + 1
                class_conf_tensor = input_data[class_conf_index]
                if class_conf_tensor.shape[0] == 1:
                    class_conf_tensor = class_conf_tensor.squeeze(0)
                classes_conf.append(class_conf_tensor)

                scores.append(np.ones_like(class_conf_tensor[:, :1, :, :], dtype=np.float32))

        # --- Common processing starts here ---

        # Helper function for flattening (can be defined here or as a class method)
        def sp_flatten(_in):
            # Input shape: (anchors_or_branches, channels, height, width)
            # Transpose to: (anchors_or_branches, height, width, channels)
            _in = _in.transpose(0, 2, 3, 1)
            # Reshape to: (anchors_or_branches * height * width, channels)
            ch = _in.shape[-1]
            return _in.reshape(-1, ch)

        # Flatten all collected data
        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores] # For v5: obj scores, For v11: ones

        # Concatenate data from all heads/branches
        boxes = np.concatenate(boxes)           # Shape: (total_detections, 4)
        classes_conf = np.concatenate(classes_conf) # Shape: (total_detections, num_classes)
        scores = np.concatenate(scores)         # Shape: (total_detections, 1)

        # filter according to threshold
        boxes, classes, scores = self._filter_boxes(boxes, scores, classes_conf)

        # Check if any boxes survived filtering
        if boxes is None or len(boxes) == 0:
            return None, None, None

        # --- Non-Maximum Suppression (NMS) per class ---
        nboxes, nclasses, nscores = [], [], []
        unique_classes = set(classes)

        for c in unique_classes:
            # Get indices for the current class
            inds = np.where(classes == c)

            # Extract data for the current class
            b = boxes[inds]
            cls = classes[inds] # Keep class info (all will be 'c')
            s = scores[inds]
            keep_indices = self._nms_boxes(b, s)

            # `keep_indices` returned by NMS are relative to the input `b`, `cls`, `s`
            if len(keep_indices) > 0:
                nboxes.append(b[keep_indices])
                nclasses.append(cls[keep_indices])
                nscores.append(s[keep_indices])

        # Check if NMS resulted in any detections
        if not nclasses: # Check if the list is empty
            return None, None, None

        # Concatenate results from all classes after NMS
        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    def _letter_box(self, im, new_shape=(640, 640), color=(0, 0, 0)):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    def _get_real_box(self, boxes, origin_shape):
        boxes[:, [0, 2]] = boxes[:, [0, 2]] - self.pad_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] - self.pad_h
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.scale_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.scale_h

        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_shape[1])
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_shape[0])
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_shape[1])
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_shape[0])
        return boxes

    def detect(self, img_src):
        """
        Detect objects in the input image.

        Args:
            img_src: Input image (numpy array).

        Returns:
            boxes: Detected bounding boxes (numpy array).
            classes: Detected classes (numpy array).
            scores: Confidence scores of the detections (numpy array).
        """
        origin_shape = img_src.shape
        img, ratio, (dw, dh) = self._letter_box(im=img_src.copy(), new_shape=(self.img_size[1], self.img_size[0]),
                                                color=(0, 0, 0))
        self.scale_w, self.scale_h = ratio
        self.pad_w, self.pad_h = dw, dh

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = img

        outputs = self.model.run([input_data])
        boxes, classes, scores = self._post_process(outputs)

        if boxes is not None:
            boxes = self._get_real_box(boxes, origin_shape)

        return boxes, classes, scores

    def draw_detections(self, image, boxes, classes, scores):
        """Draws the detection results on the image."""
        if boxes is None:
            return image
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = [int(_b) for _b in box]
            # print("%s @ (%d %d %d %d) %.3f" % (self.classes[cl], top, left, right, bottom, score))
            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(self.classes[cl], score),
                        (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return image

    def release(self):
        self.model.release()