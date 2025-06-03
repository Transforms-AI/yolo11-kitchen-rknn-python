import threading
import cv2
import time
import os
import logging
from libraries.datasend import DataUploader 
from libraries.utils import time_to_string 

# --- Information About Script ---
__author__ = "TransformsAI"
__version__ = "2.5.1" 
__status__ = "Production"
__last_updated__ = "2025-06-03" 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
logging.info(f"Script Version: {__version__}, Status: {__status__}, Last Updated: {__last_updated__}")

class VideoCaptureAsync:
    """
    Asynchronous video capture class with robust handling for different source types
    (files vs. streams), optional looping for files, and automatic restart on failure.
    """

    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

    def __init__(self, src=0, width=None, height=None, driver=None, loop=False,
                 heartbeat_config=None, auto_restart_on_fail=False, restart_delay=30.0):
        self.src = src
        self.width = width # Currently does not resize frames, but can be used for future enhancements
        self.height = height # Same as width, for potential future resizing
        self.driver = driver
        self._is_file_source = self._check_if_file_source(src)
        self.loop = loop if self._is_file_source else False

        self.auto_restart_on_fail = auto_restart_on_fail
        self.restart_delay = restart_delay

        self.cap = None
        self.started = False
        self._grabbed = False
        self._frame = None
        self._read_lock = threading.Lock()
        self._thread = None
        self._fps = 30.0
        self._last_frame_time = 0
        self._frame_count = 0
        self._stop_event = threading.Event()

        self._heartbeat_config = heartbeat_config or {}
        self._data_uploader = None
        self._last_heartbeat_time = 0

        if self._heartbeat_config.get('enabled', False):
            self._initialize_heartbeat()

        try:
            self._initialize_capture() # First attempt to initialize, should raise error if failed. Heartbeat is handled by _update
            # Didn't raise, so capture is initialized successfully.
            self._send_heartbeat(f"Video source {self.src} initialized successfully.")
        except RuntimeError as e:
            if not self.auto_restart_on_fail:
                self._send_heartbeat(f"Video source {self.src} initialization was failed, auto_restart is disabled.")
                raise
            else:
                # The _update thread (when start() is called) will handle restart attempts if auto restart is enabled.
                logging.warning(
                    f"[{self.src}] Initial capture failed: {e}. "
                    f"auto_restart_on_fail is True, will attempt restart when capture is started."
                )

    def _initialize_heartbeat(self):
        try:
            heartbeat_url = self._heartbeat_config.get('heartbeat_url')
            headers = self._heartbeat_config.get('headers', {})
            if heartbeat_url:
                self._data_uploader = DataUploader(
                    api_url=None, 
                    heartbeat_url=heartbeat_url, 
                    headers=headers,
                    debug=True,
                    max_workers=2,
                    source="Video Capture"
                )
                self._last_heartbeat_time = 0
                logging.info(f"[{self.src}] Heartbeat functionality initialized.")
            else:
                logging.warning(f"[{self.src}] Heartbeat enabled but no heartbeat_url provided.")
        except Exception as e:
            logging.error(f"[{self.src}] Failed to initialize heartbeat: {e}")
            self._data_uploader = None

    def _send_heartbeat(self, custom_message=None):
        if not self._data_uploader or not self._heartbeat_config.get('enabled', False):
            return

        current_time = time.time()
        sn = self._heartbeat_config.get('sn', f"capture_{self.src}")
        
        self._data_uploader.send_heartbeat(sn, time_to_string(current_time), status_log=custom_message)
        logging.info(f"[{sn}] Heartbeat sent: {custom_message if custom_message else 'No custom message provided'}")
        
        return

    def _check_if_file_source(self, source):
        if isinstance(source, str):
            _, ext = os.path.splitext(source)
            if ext.lower() in self.VIDEO_EXTENSIONS:
                return os.path.exists(source)
        return False

    def _initialize_capture(self):
        """
        Initializes or re-initializes the cv2.VideoCapture object.
        Raises RuntimeError on critical failure.
        Sets self._fps, self._frame_count.
        """
        try:
            if self.cap is not None: # Release existing capture if we are re-initializing
                self.cap.release()
                self.cap = None
            
            logging.debug(f"[{self.src}] Attempting to open capture source.")
            if self.driver is None:
                self.cap = cv2.VideoCapture(self.src)
            else:
                self.cap = cv2.VideoCapture(self.src, self.driver)

            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video source: {self.src}")

            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps is not None and fps > 0:
                self._fps = fps
            else:
                self._fps = 30.0 # Fallback default
                logging.warning(f"[{self.src}] Fallback FPS to {self._fps} due to detection failure.")
            
            self._frame_count = 0
            if self._is_file_source:
                frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if frame_count > 0:
                    self._frame_count = int(frame_count)
                logging.info(f"[{self.src}] Initialized video file source: "
                             f"(FPS: {self._fps:.2f}, Frames: {self._frame_count})")
            else:
                logging.info(f"[{self.src}] Initialized stream/camera source (FPS: {self._fps:.2f})")

        except Exception as e:
            # Ensure cap is None if initialization failed partway
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            raise RuntimeError(f"Capture initialization failed for {self.src}: {e}") from e

    def _attempt_restart_capture(self):
        """
        Attempts to re-initialize the video capture. Manages heartbeats for restart status.
        Returns True if restart was successful and capture is open, False otherwise.
        """
        logging.info(f"[{self.src}] Attempting to restart capture...")
        try:
            self._initialize_capture()
            if self.cap and self.cap.isOpened():
                logging.info(f"[{self.src}] Successfully restarted capture.")
                self._send_heartbeat(f"Video capture for {self.src} recovered after restart.")
                self._last_frame_time = 0 # Reset frame timer for file sources
                self._grabbed = False # Ensure read() waits for a new frame
                self._frame = None
                return True
            else:
                # This case implies _initialize_capture didn't raise but cap is not open.
                err_msg = f"Re-initialization completed, but capture {self.src} is not open (unexpected)."
                logging.error(f"[{self.src}] {err_msg}")
                self._send_heartbeat(err_msg)
                return False
        except RuntimeError as e: # Raised by _initialize_capture on hard failure
            logging.error(f"[{self.src}] Error during restart attempt: {e}")
            self._send_heartbeat(str(e)) # Send error heartbeat
            return False

    def get(self, propId):
        if self.cap:
            return self.cap.get(propId)
        return None

    def set(self, propId, value):
        if self.cap:
            return self.cap.set(propId, value)
        return False

    def start(self, loop=None):
        if self.started:
            logging.warning(f'[{self.src}] Asynchronous video capturing has already been started.')
            return self

        if not self.cap or not self.cap.isOpened():
            if not self.auto_restart_on_fail:
                # If not auto-restarting, then it's an error to start if cap is not ready.
                error_msg = (f"Capture device {self.src} not initialized or already released. "
                             f"Cannot start (auto_restart_on_fail is False).")
                logging.error(f"[{self.src}] {error_msg}")
                self._send_heartbeat(error_msg)
                return self
            else:
                # If auto_restart_on_fail is True, it's okay if cap is not open yet.
                # The _update thread will handle the initial attempt to open/restart.
                logging.info(
                    f"[{self.src}] Capture device not yet open, but auto_restart_on_fail is True. "
                    f"Proceeding to start thread for restart attempts."
                )

        if loop is not None and self._is_file_source:
            self.loop = loop

        self.started = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._update, name=f"VideoCaptureAsync_{self.src}", daemon=True)
        self._thread.start()
        logging.info(f"[{self.src}] Started video capture thread.")
        return self

    def _update(self):
        is_file = self._is_file_source
        target_frame_duration = 1.0 / self._fps if self._fps > 0 else 0

        while not self._stop_event.is_set():
            # --- Phase 1: Ensure capture is initialized and open ---
            if not self.cap or not self.cap.isOpened():
                initial_error_msg = f"Capture {self.src} not open/initialized prior to read."
                # Heartbeat for this specific state will be handled by restart logic or if restart is off

                if self.auto_restart_on_fail:
                    logging.warning(f"[{self.src}] {initial_error_msg} Attempting auto-restart.")
                    restarted_successfully = False
                    while not self._stop_event.is_set(): # Inner restart loop
                        if self._attempt_restart_capture(): # Handles its own heartbeats
                            restarted_successfully = True
                            break # Exit inner restart loop, proceed to read
                        else: # Restart attempt failed
                            if self._stop_event.is_set(): break 
                            logging.info(f"[{self.src}] Waiting {self.restart_delay}s before next restart attempt.")
                            self._stop_event.wait(self.restart_delay)
                    
                    if not restarted_successfully:
                        logging.error(f"[{self.src}] Auto-restart failed or was interrupted. Stopping thread.")
                        self._send_heartbeat(f"Failed to restart video capture for {self.src} after multiple attempts.")
                        break 
                else: # Not auto-restarting, and cap is bad
                    self._send_heartbeat(f"Auto-restart is disabled for {self.src}. Capture not open.")
                    logging.error(f"[{self.src}] {initial_error_msg} Auto-restart disabled. Stopping thread.")
                    break 
            
            # --- Phase 2: Attempt to read frame ---
            grabbed = False
            frame = None
            attempted_read_this_cycle = False # For file source timing

            try:
                if is_file:
                    current_time = time.monotonic()
                    time_since_last = current_time - self._last_frame_time
                    if time_since_last >= target_frame_duration or self._last_frame_time == 0:
                        attempted_read_this_cycle = True
                        grabbed, frame = self.cap.read()
                        if not grabbed: # End of file or read error
                            if self.loop and self._frame_count > 0:
                                logging.info(f"[{self.src}] Looping video file.")
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                grabbed, frame = self.cap.read()
                                if grabbed:
                                    self._last_frame_time = time.monotonic()
                                # If still not grabbed, it's a read failure handled below
                            else: # End of file and not looping
                                logging.info(f"[{self.src}] End of video file reached (not looping).")
                                break # Exit _update loop
                        else: # Frame grabbed successfully from file
                            self._last_frame_time = current_time
                    else: # Not enough time passed for file source
                        sleep_time = target_frame_duration - time_since_last
                        if sleep_time > 0.001:
                           self._stop_event.wait(sleep_time) # Interruptible sleep
                        continue # Skip to next iteration
                else: # Stream/Camera source
                    attempted_read_this_cycle = True
                    grabbed, frame = self.cap.read()

                # --- Phase 3: Handle read outcome ---
                if attempted_read_this_cycle:
                    if not grabbed: # Read failure (stream, or file after loop attempt)
                        error_msg = f"Failed to grab frame from source: {self.src}."
                        logging.warning(f"[{self.src}] {error_msg}")
                        self._send_heartbeat(error_msg)

                        if self.auto_restart_on_fail:
                            if self.cap: self.cap.release(); self.cap = None # Invalidate cap
                            logging.info(f"[{self.src}] Marked capture for restart due to read failure.")
                            continue # Go to top of _update loop, Phase 1 will handle restart
                        else: # Not auto-restarting after read failure
                            if not is_file: # Stream: delay and let loop retry read on next iter
                                self._stop_event.wait(0.1)
                            # For looping file that failed read and no auto-restart: error logged, will keep trying.
                            continue # Try reading again or sleep
                    else: # Frame successfully grabbed
                        with self._read_lock:
                            self._grabbed = True
                            self._frame = frame
                            

            except cv2.error as e:
                error_msg = f"OpenCV error during capture from {self.src}: {e}"
                logging.error(f"[{self.src}] {error_msg}")
                self._send_heartbeat(error_msg)
                if self.auto_restart_on_fail:
                    if self.cap: self.cap.release(); self.cap = None
                    logging.info(f"[{self.src}] Marked capture for restart due to OpenCV error.")
                    continue # Let Phase 1 handle restart
                else:
                    self._stop_event.wait(0.1) # Brief pause before retrying
                    continue
            except Exception as e:
                error_msg = f"Unexpected error in capture thread for {self.src}: {e}"
                logging.exception(f"[{self.src}] {error_msg}") # Use .exception for stack trace
                self._send_heartbeat(error_msg)
                break # Exit loop on critical unexpected errors

        self.started = False
        logging.info(f"[{self.src}] Video capture thread stopped.")
        # Send a final heartbeat indicating stopped status if it wasn't due to an error already reported
        if not self._stop_event.is_set(): # if stopped due to natural end (e.g. non-looping file)
             self._send_heartbeat(f"Video capture {self.src} thread stopped normally.")


    def read(self, wait_for_frame=False, timeout=1.0):
        if wait_for_frame and not self._grabbed and self.started: # Check self.started
            start_time = time.monotonic()
            while not self._grabbed and self.started: # Check self.started in loop
                if time.monotonic() - start_time > timeout:
                    error_msg = f"Timeout waiting for first frame from {self.src}"
                    logging.warning(f"[{self.src}] {error_msg}")
                    return False, None
                if self._stop_event.is_set(): # If thread stopped while waiting
                    return False, None
                time.sleep(0.005)

        with self._read_lock:
            frame = self._frame.copy() if self._grabbed and self._frame is not None else None
            grabbed = self._grabbed
        return grabbed, frame

    def stop(self):
        if not self.started:
            return
        logging.info(f"[{self.src}] Stopping video capture thread...")
        self._stop_event.set()

    def release(self):
        if self._data_uploader: self._data_uploader.shutdown()
        self.stop()
        if self._thread is not None and self._thread.is_alive():
             self._thread.join(timeout=max(2.0, self.restart_delay + 1.0)) # Ensure join timeout is sufficient
             if self._thread.is_alive():
                 logging.warning(f"[{self.src}] Capture thread did not stop gracefully.")

        if self.cap is not None:
            try:
                self.cap.release()
                logging.info(f"[{self.src}] Released video capture device.")
            except Exception as e:
                error_msg = f"Error releasing capture device for {self.src}: {e}"
                logging.error(f"[{self.src}] {error_msg}")
        
        if self._data_uploader:
            try:
                self._data_uploader.shutdown(wait=False) # Assuming DataUploader has shutdown
            except Exception as e:
                logging.error(f"[{self.src}] Error shutting down data uploader: {e}")
        
        self.cap = None
        self.started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
