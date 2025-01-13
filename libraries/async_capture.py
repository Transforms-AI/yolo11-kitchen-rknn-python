import threading
import cv2
import time

class VideoCaptureAsync:
    """
    Asynchronous video capture class with looping functionality.

    Attributes:
        src (int or str): The video source. Can be a camera index (0, 1, ...) or a video file path.
        width (int): The desired width of the captured frames.
        height (int): The desired height of the captured frames.
        driver (int, optional): The backend API to use for video capture (e.g., cv2.CAP_DSHOW).
        loop (bool): Whether to loop the video when it ends.
        cap (cv2.VideoCapture): The OpenCV video capture object.
        started (bool): Indicates whether the capture thread has started.
        _grabbed (bool): Indicates whether a frame was successfully grabbed in the last read.
        _frame (numpy.ndarray): The most recently captured frame.
        _read_lock (threading.Lock): A lock to ensure thread-safe access to _grabbed and _frame.
        _thread (threading.Thread): The thread responsible for capturing frames.
        _fps (float): Frames per second of the video source.
        _last_frame_time (float): Timestamp of the last grabbed frame.
        _frame_count (int): Total number of frames in the video (if applicable).
    """

    def __init__(self, src=0, width=640, height=480, driver=None, loop=False):
        """
        Initializes the VideoCaptureAsync object.

        Args:
            src (int or str): The video source.
            width (int): The desired frame width.
            height (int): The desired frame height.
            driver (int, optional): The backend API to use.
            loop (bool): Whether to loop the video.
        """
        self.src = src
        self.width = width
        self.height = height
        self.driver = driver
        self.loop = loop
        self.cap = None
        self.started = False
        self._grabbed = False
        self._frame = None
        self._read_lock = threading.Lock()
        self._thread = None
        self._fps = None
        self._last_frame_time = 0
        self._frame_count = 0

        self._initialize_capture()

    def _initialize_capture(self):
        """Initializes the cv2.VideoCapture object."""
        if self.driver is None:
            self.cap = cv2.VideoCapture(self.src)
        else:
            self.cap = cv2.VideoCapture(self.src, self.driver)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {self.src}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Get the FPS of the video source
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self._fps == 0:
            print("Warning: Could not determine FPS. Defaulting to 30.")
            self._fps = 30

        # Get the total number of frames if it's a video file
        if isinstance(self.src, str):  # Check if it's a file path
            self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get(self, propId):
        """
        Gets a property of the video capture.

        Args:
            propId (int): The property identifier (e.g., cv2.CAP_PROP_FPS).

        Returns:
            float: The value of the property.
        """
        return self.cap.get(propId)

    def set(self, propId, value):
        """
        Sets a property of the video capture.

        Args:
            propId (int): The property identifier.
            value (float or int): The new value for the property.
        """
        self.cap.set(propId, value)

    def start(self, loop=None):
        """
        Starts the asynchronous video capture thread.

        Args:
            loop (bool, optional): Whether to loop the video. Overrides the instance's loop setting.

        Returns:
            VideoCaptureAsync: The current instance.
        """
        if self.started:
            print('[!] Asynchronous video capturing has already been started.')
            return self

        if loop is not None:
            self.loop = loop

        self.started = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def _update(self):
        """
        Continuously captures frames from the video source, respecting the video's FPS and handling looping.
        This method runs in a separate thread.
        """
        while self.started:
            current_time = time.time()
            time_since_last_frame = current_time - self._last_frame_time

            # Calculate the desired time between frames based on FPS
            desired_frame_time = 1.0 / self._fps

            # If enough time has passed, grab the next frame
            if time_since_last_frame >= desired_frame_time:
                grabbed, frame = self.cap.read()

                # Handle looping if enabled and it's a video file
                if self.loop and not grabbed and self._frame_count > 0:
                    # Reset to the beginning of the video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    grabbed, frame = self.cap.read()

                with self._read_lock:
                    self._grabbed = grabbed
                    self._frame = frame
                    self._last_frame_time = current_time
            else:
                # If not enough time has passed, sleep for a short duration
                time.sleep(desired_frame_time - time_since_last_frame)

    def read(self):
        """
        Reads the most recently captured frame.

        Returns:
            tuple: A tuple containing:
                - bool: True if a frame was successfully read, False otherwise.
                - numpy.ndarray: The captured frame (or None if no frame was read).
        """
        with self._read_lock:
            grabbed = self._grabbed
            frame = self._frame.copy() if grabbed else None
        return grabbed, frame

    def stop(self):
        """
        Stops the asynchronous video capture thread.
        """
        self.started = False
        if self._thread is not None:
            self._thread.join()

    def release(self):
        """
        Releases the video capture object and stops the thread.
        """
        self.stop()
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        """
        Enters the context manager (for use with 'with' statement).
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exits the context manager and releases resources.
        """
        self.release()