import threading
import cv2
import time
import os
import logging
import socket
from libraries.datasend import DataUploader
from libraries.utils import time_to_string

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoCaptureAsync:
    """
    Asynchronous video capture class with robust handling for different source types
    (files vs. streams) and optional looping for files.

    Attributes:
        src (int or str): The video source. Can be a camera index (0, 1, ...),
                          a video file path, or a stream URL (e.g., RTSP).
        width (int): The desired width of the captured frames.
        height (int): The desired height of the captured frames.
        driver (int, optional): The backend API to use for video capture (e.g., cv2.CAP_DSHOW).
        loop (bool): Whether to loop the video *if* it's a file source. Ignored for streams.
        cap (cv2.VideoCapture): The OpenCV video capture object.
        started (bool): Indicates whether the capture thread has started.
        _is_file_source (bool): True if the source is identified as a local video file.
        _grabbed (bool): Indicates whether a frame was successfully grabbed in the last read attempt.
        _frame (numpy.ndarray or None): The most recently captured frame.
        _read_lock (threading.Lock): A lock to ensure thread-safe access to _grabbed and _frame.
        _thread (threading.Thread): The thread responsible for capturing frames.
        _fps (float): Frames per second of the video source (relevant mainly for files).
        _last_frame_time (float): Timestamp of the last grabbed frame (used for file playback timing).
        _frame_count (int): Total number of frames in the video (if applicable and determinable).
        _stop_event (threading.Event): Event to signal the thread to stop.
        _heartbeat_config (dict): Configuration for heartbeat functionality.
        _data_uploader (DataUploader): DataUploader instance for sending heartbeats.
        _last_heartbeat_time (float): Timestamp of the last heartbeat sent.
        _error_status (str): Current error status for heartbeat logging.
    """

    # Common video file extensions
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

    def __init__(self, src=0, width=640, height=480, driver=None, loop=False, 
                 heartbeat_config=None):
        """
        Initializes the VideoCaptureAsync object.

        Args:
            src (int or str): The video source (camera index, file path, stream URL).
            width (int): The desired frame width.
            height (int): The desired frame height.
            driver (int, optional): The backend API to use (e.g., cv2.CAP_DSHOW).
            loop (bool): Whether to loop the video *if* it's a file source.
            heartbeat_config (dict, optional): Configuration for heartbeat functionality.
                Expected keys:
                - 'enabled' (bool): Whether to enable heartbeat functionality
                - 'sn' (str): Serial number for identification
                - 'heartbeat_url' (str): URL for sending heartbeats
                - 'headers' (dict): Headers for heartbeat requests
                - 'interval' (float): Heartbeat interval in seconds (default: 30)
        """
        self.src = src
        self.width = width
        self.height = height
        self.driver = driver
        self._is_file_source = self._check_if_file_source(src)
        self.loop = loop if self._is_file_source else False # Looping only makes sense for files

        self.cap = None
        self.started = False
        self._grabbed = False
        self._frame = None
        self._read_lock = threading.Lock()
        self._thread = None
        self._fps = 30.0  # Default FPS
        self._last_frame_time = 0
        self._frame_count = 0
        self._stop_event = threading.Event()

        # Heartbeat functionality
        self._heartbeat_config = heartbeat_config or {}
        self._data_uploader = None
        self._last_heartbeat_time = 0
        self._error_status = ""
        
        if self._heartbeat_config.get('enabled', False):
            self._initialize_heartbeat()

        self._initialize_capture()

    def _initialize_heartbeat(self):
        """Initialize the heartbeat functionality."""
        try:
            heartbeat_url = self._heartbeat_config.get('heartbeat_url')
            headers = self._heartbeat_config.get('headers', {})
            
            if heartbeat_url:
                self._data_uploader = DataUploader(
                    api_url=None,  # We only need heartbeat functionality
                    heartbeat_url=heartbeat_url,
                    headers=headers,
                    print_response=False,
                    debug=False
                )
                self._last_heartbeat_time = 0
                logging.info(f"Heartbeat functionality initialized for source: {self.src}")
            else:
                logging.warning(f"Heartbeat enabled but no heartbeat_url provided for source: {self.src}")
        except Exception as e:
            logging.error(f"Failed to initialize heartbeat for source {self.src}: {e}")
            self._data_uploader = None

    def _send_heartbeat_if_needed(self, force_send=False, error_message=""):
        """Send heartbeat if interval has passed or if forced."""
        if not self._data_uploader or not self._heartbeat_config.get('enabled', False):
            return
            
        current_time = time.time()
        heartbeat_interval = self._heartbeat_config.get('interval', 30)
        
        if force_send or (current_time - self._last_heartbeat_time >= heartbeat_interval):
            try:
                sn = self._heartbeat_config.get('sn', f"capture_{self.src}")
                status_log = error_message if error_message else self._error_status or "Video capture operational"
                
                self._data_uploader.send_heartbeat(sn, time_to_string(current_time), status_log=status_log)
                self._last_heartbeat_time = time.time()
                
                if error_message:
                    logging.info(f"[{sn}] Error heartbeat sent: {error_message}")
                else:
                    logging.debug(f"[{sn}] Regular heartbeat sent")
                    
            except Exception as e:
                logging.error(f"Failed to send heartbeat for source {self.src}: {e}")

    def _check_if_file_source(self, source):
        """Checks if the source is likely a local video file."""
        if isinstance(source, str):
            # Check if it has a common video file extension
            _, ext = os.path.splitext(source)
            if ext.lower() in self.VIDEO_EXTENSIONS:
                # Further check if the file actually exists locally
                # This helps differentiate between local files and URLs ending in .mp4 etc.
                # Note: This might not be foolproof for all network paths but covers common cases.
                return os.path.exists(source)
        return False # Integers (camera indices) or non-file strings are not files

    def _initialize_capture(self):
        """Initializes the cv2.VideoCapture object."""
        try:
            if self.driver is None:
                self.cap = cv2.VideoCapture(self.src)
            else:
                self.cap = cv2.VideoCapture(self.src, self.driver)

            if not self.cap.isOpened():
                error_msg = f"Could not open video source: {self.src}"
                self._error_status = error_msg
                # self._send_heartbeat_if_needed(force_send=True, error_message=error_msg)
                raise ValueError(error_msg)

            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # Attempt to get FPS, provide a more informative warning if it fails
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps is not None and fps > 0:
                self._fps = fps
            else:
                warning_msg = (f"Could not determine FPS for source: {self.src}. "
                              f"Defaulting to {self._fps} FPS. "
                              f"Frame timing might be inaccurate for file sources.")
                logging.warning(warning_msg)
                self._error_status = warning_msg
                self._send_heartbeat_if_needed(force_send=False, error_message=warning_msg)

            # Get frame count only if it's identified as a file source
            if self._is_file_source:
                frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if frame_count > 0:
                    self._frame_count = int(frame_count)
                logging.info(f"Initialized video file source: {self.src} "
                             f"(FPS: {self._fps:.2f}, Frames: {self._frame_count})")
            else:
                logging.info(f"Initialized stream/camera source: {self.src} (FPS: {self._fps:.2f})")

            # Clear error status on successful initialization
            self._error_status = ""

        except Exception as e:
            error_msg = f"[ERROR] {e}"
            self._error_status = error_msg
            self._send_heartbeat_if_needed(force_send=True, error_message=error_msg)
            time.sleep(5)
            self.release() # Ensure resources are cleaned up on initialization failure
            raise RuntimeError(error_msg) from e

    def get(self, propId):
        """
        Gets a property of the video capture. Thread-safe.

        Args:
            propId (int): The property identifier (e.g., cv2.CAP_PROP_FPS).

        Returns:
            float: The value of the property, or None if capture is not initialized.
        """
        if self.cap:
            return self.cap.get(propId)
        return None

    def set(self, propId, value):
        """
        Sets a property of the video capture. Thread-safe.

        Args:
            propId (int): The property identifier.
            value (float or int): The new value for the property.

        Returns:
            bool: True if the property was set successfully, False otherwise.
        """
        if self.cap:
            return self.cap.set(propId, value)
        return False

    def start(self, loop=None):
        """
        Starts the asynchronous video capture thread.

        Args:
            loop (bool, optional): Overrides the instance's loop setting
                                   (only effective if it's a file source).

        Returns:
            VideoCaptureAsync: The current instance.
        """
        if self.started:
            logging.warning('Asynchronous video capturing has already been started.')
            return self

        if not self.cap or not self.cap.isOpened():
            error_msg = "Capture device not initialized or already released. Cannot start."
            logging.error(error_msg)
            self._error_status = error_msg
            self._send_heartbeat_if_needed(force_send=False, error_message=error_msg)
            return self

        # Update loop setting if provided and applicable
        if loop is not None and self._is_file_source:
            self.loop = loop

        self.started = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._update, name=f"VideoCaptureAsync_{self.src}", daemon=True)
        self._thread.start()
        logging.info(f"Started video capture thread for {self.src}")
        return self

    def _update(self):
        """
        Continuously captures frames from the video source.
        Runs in a separate thread. Handles file sources with timing/looping
        and stream sources by reading as fast as possible.
        """
        is_file = self._is_file_source
        target_frame_duration = 1.0 / self._fps if self._fps > 0 else 0

        while not self._stop_event.is_set():
            if not self.cap or not self.cap.isOpened():
                error_msg = f"Capture source {self.src} closed unexpectedly. Stopping thread."
                logging.error(error_msg)
                self._error_status = error_msg
                self._send_heartbeat_if_needed(force_send=True, error_message=error_msg)
                break

            grabbed = False
            frame = None

            try:
                if is_file:
                    # --- File Source Logic ---
                    current_time = time.monotonic() # Use monotonic clock for intervals
                    time_since_last = current_time - self._last_frame_time

                    if time_since_last >= target_frame_duration or self._last_frame_time == 0:
                        grabbed, frame = self.cap.read()

                        if not grabbed:
                            # End of file reached
                            if self.loop and self._frame_count > 0:
                                logging.info(f"Looping video file: {self.src}")
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset to beginning
                                grabbed, frame = self.cap.read() # Read the first frame again
                                if grabbed:
                                     self._last_frame_time = time.monotonic() # Reset timer only if read succeeds
                                else:
                                     error_msg = f"Failed to read frame after looping {self.src}"
                                     logging.warning(error_msg)
                                     self._error_status = error_msg
                                     self._send_heartbeat_if_needed(error_message=error_msg)
                                     # Keep grabbed as False, loop will try again or exit
                            else:
                                # End of file and not looping, stop reading
                                logging.info(f"End of video file reached: {self.src}")
                                break # Exit the update loop
                        else:
                             self._last_frame_time = current_time # Update time only when frame grabbed

                        # Update shared variables only if read attempt was made
                        with self._read_lock:
                            self._grabbed = grabbed
                            self._frame = frame
                    else:
                        # Not enough time passed, sleep briefly
                        sleep_time = target_frame_duration - time_since_last
                        # Avoid sleeping for negative or zero duration
                        if sleep_time > 0.001: # Sleep only for meaningful durations
                           time.sleep(sleep_time)
                        continue # Skip updating shared vars, go to next loop iteration

                else:
                    # --- Stream/Camera Source Logic ---
                    # Read as fast as possible, cap.read() will block if needed
                    grabbed, frame = self.cap.read()
                    if not grabbed:
                        # Stream might have ended or encountered an error
                        error_msg = f"Failed to grab frame from stream/camera: {self.src}. Retrying..."
                        logging.warning(error_msg)
                        self._error_status = error_msg
                        self._send_heartbeat_if_needed(error_message=error_msg)
                        # Add a small delay to prevent tight loop on persistent errors
                        time.sleep(0.01)
                        # Keep trying unless stopped

                    # Update shared variables immediately after read attempt
                    with self._read_lock:
                        self._grabbed = grabbed
                        self._frame = frame

                # Send regular heartbeat if needed (when no errors)
                if grabbed and not self._error_status:
                    self._send_heartbeat_if_needed()
                
                # Clear error status if we successfully grabbed a frame
                if grabbed and self._error_status:
                    self._error_status = ""
                    self._send_heartbeat_if_needed(force_send=True, error_message="Video capture recovered")

            except cv2.error as e:
                error_msg = f"OpenCV error during capture from {self.src}: {e}"
                logging.error(error_msg)
                self._error_status = error_msg
                self._send_heartbeat_if_needed(force_send=False, error_message=error_msg)
                # Decide how to handle: maybe stop, maybe just log and continue
                # For now, log and set grabbed=False, let the loop continue/retry
                with self._read_lock:
                    self._grabbed = False
                    self._frame = None
                time.sleep(0.1) # Prevent spamming logs on persistent error
            except Exception as e:
                error_msg = f"Unexpected error in capture thread for {self.src}: {e}"
                logging.exception(error_msg)
                self._error_status = error_msg
                self._send_heartbeat_if_needed(force_send=False, error_message=error_msg)
                with self._read_lock:
                    self._grabbed = False
                    self._frame = None
                break # Exit loop on unexpected errors

        # End of loop (either stopped or error)
        self.started = False
        logging.info(f"Video capture thread stopped for {self.src}")


    def read(self, wait_for_frame=False, timeout=1.0):
        """
        Reads the most recently captured frame.

        Args:
            wait_for_frame (bool): If True, wait until a new frame is available
                                   or timeout occurs. Useful for initial frame.
            timeout (float): Maximum time in seconds to wait if wait_for_frame is True.

        Returns:
            tuple: A tuple containing:
                - bool: True if a frame was successfully read, False otherwise.
                - numpy.ndarray or None: The captured frame.
        """
        if wait_for_frame and not self._grabbed:
            start_time = time.monotonic()
            while not self._grabbed and self.started:
                if time.monotonic() - start_time > timeout:
                    error_msg = f"Timeout waiting for first frame from {self.src}"
                    logging.warning(error_msg)
                    self._error_status = error_msg
                    self._send_heartbeat_if_needed(error_message=error_msg)
                    return False, None
                time.sleep(0.005) # Small sleep to yield execution

        with self._read_lock:
            # Make a copy only if the frame is not None to avoid errors
            frame = self._frame.copy() if self._grabbed and self._frame is not None else None
            grabbed = self._grabbed
        return grabbed, frame

    def stop(self):
        """
        Signals the asynchronous video capture thread to stop.
        """
        if not self.started:
            return

        logging.info(f"Stopping video capture thread for {self.src}...")
        self._stop_event.set()

    def release(self):
        """
        Stops the thread and releases the video capture object.
        """
        self.stop()
        # Wait for the thread to finish
        if self._thread is not None and self._thread.is_alive():
             # Add a timeout to join to prevent indefinite blocking
             self._thread.join(timeout=2.0)
             if self._thread.is_alive():
                 logging.warning(f"Capture thread for {self.src} did not stop gracefully.")

        if self.cap is not None:
            try:
                self.cap.release()
                logging.info(f"Released video capture device for {self.src}")
            except Exception as e:
                error_msg = f"Error releasing capture device for {self.src}: {e}"
                logging.error(error_msg)
                self._error_status = error_msg
                # self._send_heartbeat_if_needed(force_send=False, error_message=error_msg)
        
        # Shutdown data uploader if initialized
        if self._data_uploader:
            try:
                self._data_uploader.shutdown(wait=False)
            except Exception as e:
                logging.error(f"Error shutting down data uploader for {self.src}: {e}")
        
        self.cap = None
        self.started = False # Ensure started is False after release

    def __enter__(self):
        """
        Enters the context manager (for use with 'with' statement).
        Starts the capture thread.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exits the context manager and releases resources.
        """
        self.release()