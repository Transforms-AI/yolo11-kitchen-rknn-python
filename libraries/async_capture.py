import threading
import cv2
import time
import os
import logging
from libraries.datasend import DataUploader 
from libraries.utils import time_to_string 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')

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
        self._error_status = "" 

        if self._heartbeat_config.get('enabled', False):
            self._initialize_heartbeat()

        try:
            self._initialize_capture() # First attempt to initialize, should raise error if failed. Heartbeat is handled by _update
            
            # Didn't raise, so capture is initialized successfully.
            self._manage_heartbeat(recovered=True, force_send=True, 
                                    custom_message=f"Video source {self.src} initialized successfully.")
        except RuntimeError as e:
            self._error_status = str(e) # Set error status from exception
            # self._manage_heartbeat(force_send=True) # Send error heartbeat
            if not self.auto_restart_on_fail:
                self._manage_heartbeat(recovered=True, 
                                       force_send=True, 
                                       custom_message=f"Video source {self.src} initialization was failed, auto_restart is disabled.")
                raise
            else:
                # If auto_restart_on_fail is True, log the initial failure.
                # The object will be created, but self.cap will be None.
                # The _update thread (when start() is called) will handle restart attempts.
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

    def _manage_heartbeat(self, new_error_message=None, force_send=False, recovered=False, custom_message=None):
        if not self._data_uploader or not self._heartbeat_config.get('enabled', False):
            return

        current_time = time.time()
        heartbeat_interval = self._heartbeat_config.get('interval', 30)
        sn = self._heartbeat_config.get('sn', f"capture_{self.src}")

        status_changed = False
        previous_error_status = self._error_status

        if new_error_message is not None:
            if self._error_status != new_error_message:
                self._error_status = new_error_message
                status_changed = True
        elif recovered:
            if self._error_status: # Only consider it recovered if there was an error
                self._error_status = ""
                status_changed = True
        
        if status_changed:
             logging.info(f"[{sn}] Status update. Old: '{previous_error_status}', New: '{self._error_status}'")

        should_send = force_send or status_changed or \
                      (current_time - self._last_heartbeat_time >= heartbeat_interval)

        if should_send:
            try:
                if custom_message:
                    status_log_msg = custom_message
                elif self._error_status:
                    status_log_msg = self._error_status
                elif recovered: # Use if custom_message is None but recovered is True
                    status_log_msg = f"Video capture for {self.src} recovered"
                else:
                    status_log_msg = f"Video capture for {self.src} operational"
                
                self._data_uploader.send_heartbeat(sn, time_to_string(current_time), status_log=status_log_msg)
                self._last_heartbeat_time = current_time
                
                log_level = logging.DEBUG
                log_prefix = "Regular"
                if status_changed or force_send:
                    log_level = logging.INFO
                    if self._error_status and not custom_message and not recovered:
                        log_prefix = "Error"
                    elif recovered and not custom_message:
                        log_prefix = "Recovery"
                    elif custom_message: # Custom messages often indicate significant events
                        log_prefix = "Status" 
                
                logging.log(log_level, f"[{sn}] {log_prefix} heartbeat sent: {status_log_msg}")
                    
            except Exception as e:
                logging.error(f"[{sn}] Failed to send heartbeat: {e}")

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
        Sets self._error_status for non-critical warnings (e.g., FPS detection).
        Clears self._error_status on full success if no warnings.
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
            current_fps_warning = ""
            if fps is not None and fps > 0:
                self._fps = fps
            else:
                self._fps = 30.0 # Fallback default
                current_fps_warning = (f"Could not determine FPS for source: {self.src}. "
                                      f"Defaulting to {self._fps} FPS. "
                                      f"Frame timing might be inaccurate for file sources.")
                logging.warning(f"[{self.src}] {current_fps_warning}")
            
            self._frame_count = 0
            if self._is_file_source:
                frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if frame_count > 0:
                    self._frame_count = int(frame_count)
                logging.info(f"[{self.src}] Initialized video file source: "
                             f"(FPS: {self._fps:.2f}, Frames: {self._frame_count})")
            else:
                logging.info(f"[{self.src}] Initialized stream/camera source (FPS: {self._fps:.2f})")

            # Set error status if there's a warning, otherwise clear it.
            self._error_status = current_fps_warning # Will be "" if no warning

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
            self._initialize_capture() # This might set self._error_status for FPS warnings

            if self.cap and self.cap.isOpened():
                logging.info(f"[{self.src}] Successfully restarted capture.")
                if self._error_status: # FPS warning might be present
                    self._manage_heartbeat(custom_message=f"Video capture for {self.src} restarted with warning: {self._error_status}")
                else: # Clean restart
                    self._manage_heartbeat(recovered=True, force_send=True,
                                           custom_message=f"Video capture for {self.src} recovered after restart.")
                self._last_frame_time = 0 # Reset frame timer for file sources
                self._grabbed = False # Ensure read() waits for a new frame
                self._frame = None
                return True
            else:
                # This case implies _initialize_capture didn't raise but cap is not open.
                # _initialize_capture should set self._error_status or raise.
                err_msg = f"Re-initialization completed, but capture {self.src} is not open (unexpected)."
                logging.error(f"[{self.src}] {err_msg}")
                self._manage_heartbeat(new_error_message=err_msg, force_send=True)
                return False
        except RuntimeError as e: # Raised by _initialize_capture on hard failure
            logging.error(f"[{self.src}] Error during restart attempt: {e}")
            self._error_status = str(e) # Update status from exception
            self._manage_heartbeat(force_send=True) # Send error heartbeat
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
                self._manage_heartbeat(new_error_message=error_msg, force_send=True)
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
                        # Ensure a final error heartbeat if not already sent by _attempt_restart_capture
                        if not self._error_status or "failed" not in self._error_status.lower():
                             self._manage_heartbeat(new_error_message=f"Auto-restart failed for {self.src}", force_send=True)
                        break # Exit _update loop
                    # If restarted_successfully, cap is now OK. Fall through to read.
                else: # Not auto-restarting, and cap is bad
                    self._manage_heartbeat(new_error_message=initial_error_msg + " Auto-restart disabled.", force_send=True)
                    logging.error(f"[{self.src}] {initial_error_msg} Auto-restart disabled. Stopping thread.")
                    break # Exit _update loop
            
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
                                self._manage_heartbeat() # Send regular if due
                                break # Exit _update loop
                        else: # Frame grabbed successfully from file
                            self._last_frame_time = current_time
                    else: # Not enough time passed for file source
                        sleep_time = target_frame_duration - time_since_last
                        if sleep_time > 0.001:
                           self._stop_event.wait(sleep_time) # Interruptible sleep
                        self._manage_heartbeat() # Check for periodic heartbeat
                        continue # Skip to next iteration
                else: # Stream/Camera source
                    attempted_read_this_cycle = True
                    grabbed, frame = self.cap.read()

                # --- Phase 3: Handle read outcome ---
                if attempted_read_this_cycle:
                    if not grabbed: # Read failure (stream, or file after loop attempt)
                        error_msg = f"Failed to grab frame from source: {self.src}."
                        logging.warning(f"[{self.src}] {error_msg}")
                        self._manage_heartbeat(new_error_message=error_msg, force_send=True)

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
                        
                        if self._error_status: # Was an error, now recovered by successful read
                            self._manage_heartbeat(recovered=True, force_send=True,
                                                   custom_message=f"Video capture for {self.src} recovered.")
                        # else:
                            # self._manage_heartbeat() # Regular operational heartbeat, not required
                            

            except cv2.error as e:
                error_msg = f"OpenCV error during capture from {self.src}: {e}"
                logging.error(f"[{self.src}] {error_msg}")
                self._manage_heartbeat(new_error_message=error_msg, force_send=True)
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
                self._manage_heartbeat(new_error_message=error_msg, force_send=True)
                break # Exit loop on critical unexpected errors

        self.started = False
        logging.info(f"[{self.src}] Video capture thread stopped.")
        # Send a final heartbeat indicating stopped status if it wasn't due to an error already reported
        if not self._error_status and not self._stop_event.is_set(): # if stopped due to natural end (e.g. non-looping file)
             self._manage_heartbeat(custom_message=f"Video capture {self.src} thread stopped normally.", force_send=True)
        elif self._stop_event.is_set() and not self._error_status: # Explicitly stopped by user/system
             self._manage_heartbeat(custom_message=f"Video capture {self.src} thread explicitly stopped.", force_send=True)


    def read(self, wait_for_frame=False, timeout=1.0):
        if wait_for_frame and not self._grabbed and self.started: # Check self.started
            start_time = time.monotonic()
            while not self._grabbed and self.started: # Check self.started in loop
                if time.monotonic() - start_time > timeout:
                    error_msg = f"Timeout waiting for first frame from {self.src}"
                    logging.warning(f"[{self.src}] {error_msg}")
                    # Don't set self._error_status from read timeout, it's a consumer issue
                    # self._manage_heartbeat(new_error_message=error_msg) # Optional: report this via heartbeat
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
                # self._manage_heartbeat(new_error_message=error_msg) # Heartbeat system might be down
        
        if self._data_uploader:
            try:
                # Send final heartbeat if possible, indicating shutdown
                final_msg = self._error_status if self._error_status else f"Video capture {self.src} released."
                self._manage_heartbeat(custom_message=final_msg, force_send=True)
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