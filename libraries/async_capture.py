import threading
import cv2

class VideoCaptureAsync:
    """
    Asynchronous video capture class.

    Attributes:
        src (int or str): The video source. Can be a camera index (0, 1, ...) or a video file path.
        width (int): The desired width of the captured frames.
        height (int): The desired height of the captured frames.
        driver (int, optional): The backend API to use for video capture (e.g., cv2.CAP_DSHOW).
        cap (cv2.VideoCapture): The OpenCV video capture object.
        started (bool): Indicates whether the capture thread has started.
        _grabbed (bool): Indicates whether a frame was successfully grabbed in the last read.
        _frame (numpy.ndarray): The most recently captured frame.
        _read_lock (threading.Lock): A lock to ensure thread-safe access to _grabbed and _frame.
        _thread (threading.Thread): The thread responsible for capturing frames.
    """

    def __init__(self, src=0, width=640, height=480, driver=None):
        """
        Initializes the VideoCaptureAsync object.

        Args:
            src (int or str): The video source.
            width (int): The desired frame width.
            height (int): The desired frame height.
            driver (int, optional): The backend API to use.
        """
        self.src = src
        self.width = width
        self.height = height
        self.driver = driver
        self.cap = None
        self.started = False
        self._grabbed = False
        self._frame = None
        self._read_lock = threading.Lock()
        self._thread = None

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

    def start(self):
        """
        Starts the asynchronous video capture thread.

        Returns:
            VideoCaptureAsync: The current instance.
        """
        if self.started:
            print('[!] Asynchronous video capturing has already been started.')
            return self

        self.started = True
        self._thread = threading.Thread(target=self._update, daemon=True)  # Use daemon thread
        self._thread.start()
        return self

    def _update(self):
        """
        Continuously captures frames from the video source.
        This method runs in a separate thread.
        """
        while self.started:
            grabbed, frame = self.cap.read()
            with self._read_lock:
                self._grabbed = grabbed
                self._frame = frame

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
            frame = self._frame.copy() if grabbed else None # Copy to avoid issues if the frame is modified outside
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