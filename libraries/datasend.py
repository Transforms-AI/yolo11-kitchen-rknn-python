import csv
import requests
import time
import os
import socket
import fcntl
import struct
import threading # threading.RLock will be used
import random
from concurrent.futures import ThreadPoolExecutor
import getpass
import json
import io # For BytesIO and type checking
import uuid # For unique filenames

class DataUploader:
    def __init__(self, api_url, heartbeat_url=None, headers=None,
                 max_workers=5, max_retries=5, retry_delay=1, timeout=10,
                 print_response=True, debug=True,
                 cache_file_path="uploader_cache.json",
                 cache_files_dir="uploader_cached_files",
                 max_cache_retries=3,
                 cache_retry_interval=100, # Seconds for periodic cache processing
                 max_cache_items=300,       # Max number of items in cache
                 max_cache_age_seconds=24*60*60 # Max age of a cache item (1 day)
                 ):
        """
        Initializes the DataUploader.
        Args:
            api_url (str): The API endpoint URL for data uploads.
            heartbeat_url (str, optional): The API endpoint URL for heartbeats.
            headers (dict, optional): Custom headers for requests.
            max_workers (int): Maximum number of worker threads for sending data.
            max_retries (int): Maximum retries for a single send attempt (before caching).
            retry_delay (int): Base delay in seconds for retries (exponential backoff).
            timeout (int): Overall timeout in seconds for a send operation (including retries).
                           Note: This is an overall timeout for the _send_data_thread operation.
                           Individual requests within it do not currently use this directly but
                           will be interrupted if the overall operation exceeds this.
            print_response (bool): Whether to print responses and status messages.
            debug (bool): Whether to print detailed debug information.
            cache_file_path (str): Path to the JSON file for storing cache metadata.
            cache_files_dir (str): Directory to store temporary files for cached uploads.
            max_cache_retries (int): Maximum times a cached item will be retried.
            cache_retry_interval (int): Interval in seconds for periodic cache processing.
                                        Set to 0 or negative to disable periodic retries.
            max_cache_items (int): Maximum number of items to keep in the cache.
                                   Older items are pruned if the limit is exceeded. 0 for no limit.
            max_cache_age_seconds (int): Maximum age in seconds for an item in the cache.
                                         Older items are pruned. 0 for no limit.
        """
        self.api_url = api_url
        self.heartbeat_url = heartbeat_url
        self.headers = headers if headers is not None else {}

        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout # Overall operation timeout for _send_data_thread
        self.print_response = print_response
        self.debug = debug

        self.mac_address = self._get_mac_address()
        self.ip_address = self._get_ip_address()
        self.hostname = os.uname().nodename
        try:
            self.os_username = os.getlogin()
        except OSError:
            self.os_username = getpass.getuser()

        self.cache_file_path = cache_file_path
        self.cache_files_dir = cache_files_dir
        self.max_cache_retries = max_cache_retries
        self.failed_sends_cache = [] # In-memory list of cache items
        # Use RLock to allow re-entrant lock acquisition by the same thread.
        # This is necessary because some methods holding the lock call other methods
        # that also attempt to acquire the same lock (e.g., _add_to_cache -> _enforce_cache_limits,
        # or _retry_failed_sends -> _remove_from_cache_by_item).
        self.cache_lock = threading.RLock()

        self.cache_retry_interval = cache_retry_interval
        self.max_cache_items = max_cache_items
        self.max_cache_age_seconds = max_cache_age_seconds
        self.cache_management_timer = None
        self.shutting_down_event = threading.Event()

        try:
            os.makedirs(self.cache_files_dir, exist_ok=True)
            if self.print_response or self.debug:
                print(f"[INFO] Ensured cache files directory exists: {self.cache_files_dir}")
        except OSError as e:
            print(f"[ERROR] Could not create cache files directory {self.cache_files_dir}: {e}. File caching will be impaired.")
            self.cache_files_dir = None # Disable file part of caching if dir creation fails

        self._load_cache() # Load existing cache metadata
        if self.cache_files_dir: # Only run orphan cleanup if directory is valid
            self._cleanup_orphaned_cached_files() # Clean up files not in metadata

        self._start_cache_management_timer() # Start periodic cache processing

    def _get_ip_address(self):
        def get_ip(ifname):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                ifname_bytes = ifname[:15].encode('utf-8')
                ifname_packed = struct.pack('256s', ifname_bytes)
                return socket.inet_ntoa(fcntl.ioctl(
                    s.fileno(),
                    0x8915,  # SIOCGIFADDR
                    ifname_packed
                )[20:24])
            except OSError:
                return None
        for iface in ['eth0', 'wlan0', 'enp0s3', 'enp1s0']: # Added common enpXsX names
            if os.path.exists(f'/sys/class/net/{iface}/operstate'):
                try:
                    with open(f'/sys/class/net/{iface}/operstate') as f:
                        state = f.read().strip()
                    if state == 'up':
                        ip = get_ip(iface)
                        if ip and ip != '127.0.0.1':
                            return ip
                except Exception:
                    continue
        try: # Fallback for other systems / no permission to /sys
            return socket.gethostbyname(socket.gethostname())
        except socket.gaierror:
            return "127.0.0.1"


    def _get_mac_address(self):
        # (Implementation remains the same as your previous version)
        try:
            for iface_name in ['eth0', 'wlan0', 'en0', 'enp0s3', 'enp1s0']: # Added common enpXsX names
                path = f'/sys/class/net/{iface_name}/address'
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        mac_address = f.read().strip()
                    if mac_address and mac_address != "00:00:00:00:00:00":
                        return mac_address
        except Exception as e:
            if self.debug: print(f"[DEBUG] Error reading MAC from /sys: {e}")

        if self.print_response or self.debug: print("[INFO] MAC address: Falling back to uuid.getnode().")
        try:
            mac_int = uuid.getnode()
            if (mac_int >> 40) % 2 == 0: # Check for locally administered bit, prefer global
                 if 0 < mac_int < (1 << 48) -1 : # Basic check for valid range
                    hex_mac = format(mac_int, '012x')
                    return ':'.join(hex_mac[i:i+2] for i in range(0, 12, 2))
            # If locally administered or out of typical range, log warning but still try to format
            hex_mac = format(mac_int, '012x')
            formatted_mac = ':'.join(hex_mac[i:i+2] for i in range(0, 12, 2))
            if self.print_response or self.debug:
                print(f"[WARNING] uuid.getnode() returned {mac_int} (formatted: {formatted_mac}), which might be a locally administered or non-ideal MAC.")
            return formatted_mac
        except Exception as e:
            if self.print_response or self.debug: print(f"[WARNING] Error using uuid.getnode() for MAC: {e}")
        return "00:00:00:00:00:00" # Ultimate fallback

    def _load_cache(self):
        with self.cache_lock:
            if os.path.exists(self.cache_file_path):
                try:
                    with open(self.cache_file_path, 'r') as f:
                        self.failed_sends_cache = json.load(f)
                    if self.print_response or self.debug:
                        print(f"[INFO] Loaded {len(self.failed_sends_cache)} items from cache: {self.cache_file_path}")
                except json.JSONDecodeError:
                    if self.print_response or self.debug:
                        print(f"[ERROR] Could not decode cache file: {self.cache_file_path}. Starting fresh.")
                    self.failed_sends_cache = []
                except Exception as e:
                    if self.print_response or self.debug:
                        print(f"[ERROR] Failed to load cache: {e}")
                    self.failed_sends_cache = []
            else:
                if self.debug: print(f"[DEBUG] Cache file {self.cache_file_path} not found. Starting fresh.")
                self.failed_sends_cache = []

    def _save_cache(self):
        """Saves the current in-memory cache to disk. Returns True on success, False on failure.
           This method assumes the cache_lock is ALREADY HELD by the caller if necessary
           for atomicity with modifications to self.failed_sends_cache.
           However, this method itself does not acquire/release the lock.
        """
        try:
            temp_cache_file_path = self.cache_file_path + ".tmp"
            cache_dir = os.path.dirname(self.cache_file_path)
            if cache_dir and not os.path.exists(cache_dir):
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                except OSError as e_mkdir:
                    if self.debug: print(f"[DEBUG] Could not create cache directory {cache_dir}: {e_mkdir}")
                    # Proceeding, open might fail if dir creation actually failed and was needed.
            with open(temp_cache_file_path, 'w') as f:
                json.dump(self.failed_sends_cache, f, indent=2)
            os.replace(temp_cache_file_path, self.cache_file_path)
            if self.debug: print(f"[DEBUG] Cache saved successfully to {self.cache_file_path}. Size: {len(self.failed_sends_cache)}")
            return True
        except Exception as e:
            # import traceback # Keep this commented out for production, uncomment for deep debugging
            # traceback.print_exc()
            if self.print_response or self.debug:
                print(f"[ERROR] Could not save cache to {self.cache_file_path}: {e}")
            if os.path.exists(temp_cache_file_path):
                try: os.remove(temp_cache_file_path)
                except OSError: pass
            return False

    def _remove_files_for_cached_item(self, item_to_clean):
        """Helper to delete physical files associated with a cache item. Does NOT modify the cache list itself.
           Assumes cache_lock is held if called during cache modification.
        """
        if item_to_clean and item_to_clean.get("cached_files") and self.cache_files_dir:
            if self.debug:
                print(f"[DEBUG] Cleaning up disk files for item: {item_to_clean.get('original_identifier')}")
            for _form_field, file_info in item_to_clean["cached_files"].items():
                cached_file_path_to_delete = file_info.get("cached_filepath")
                if cached_file_path_to_delete and os.path.exists(cached_file_path_to_delete):
                    try:
                        os.remove(cached_file_path_to_delete)
                        if self.debug:
                            print(f"  - Deleted cached file from disk: {cached_file_path_to_delete}")
                    except Exception as e:
                        if self.print_response or self.debug:
                            print(f"  - [ERROR] Failed to delete cached file {cached_file_path_to_delete} from disk: {e}")

    def _add_to_cache(self, data_payload, url_to_use, files_tuple_dict, original_identifier, is_heartbeat_type, messages_list_for_warning):
        with self.cache_lock: # Acquire lock for modifying cache and saving
            cached_files_info = {}
            temp_files_written_paths = []

            if files_tuple_dict and self.cache_files_dir:
                for form_field_name, file_tuple in files_tuple_dict.items():
                    original_filename, file_bytes_io_or_raw_bytes, mimetype = file_tuple
                    bytes_content = None
                    if isinstance(file_bytes_io_or_raw_bytes, io.BytesIO):
                        file_bytes_io_or_raw_bytes.seek(0)
                        bytes_content = file_bytes_io_or_raw_bytes.read()
                    elif isinstance(file_bytes_io_or_raw_bytes, bytes):
                        bytes_content = file_bytes_io_or_raw_bytes
                    else:
                        err_msg = (f"[ERROR] Caching: Unsupported file content type for field '{form_field_name}' "
                                   f"in '{original_identifier}'. Expected BytesIO or bytes.")
                        messages_list_for_warning.append(err_msg)
                        if self.print_response or self.debug: print(err_msg)
                        continue

                    if bytes_content is not None:
                        try:
                            safe_original_filename = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in original_filename)
                            temp_filename = f"{uuid.uuid4()}_{safe_original_filename}"
                            cached_filepath = os.path.join(self.cache_files_dir, temp_filename)
                            with open(cached_filepath, 'wb') as temp_f:
                                temp_f.write(bytes_content)
                            temp_files_written_paths.append(cached_filepath)
                            cached_files_info[form_field_name] = {
                                "original_filename": original_filename,
                                "mimetype": mimetype,
                                "cached_filepath": cached_filepath
                            }
                            msg = f"[INFO] Caching: Saved file for field '{form_field_name}' to {cached_filepath}"
                            messages_list_for_warning.append(msg)
                            if self.debug: print(msg)
                        except IOError as e:
                            err_msg = (f"[ERROR] Caching: Could not write temporary file for field '{form_field_name}' "
                                       f"in '{original_identifier}': {e}")
                            messages_list_for_warning.append(err_msg)
                            if self.print_response or self.debug: print(err_msg)
            elif files_tuple_dict and not self.cache_files_dir:
                 err_msg = f"[WARNING] Caching: Files were present for '{original_identifier}' but cache_files_dir is not configured/valid. Files not cached."
                 messages_list_for_warning.append(err_msg)
                 if self.print_response or self.debug: print(err_msg)

            cache_item = {
                "data_payload": data_payload, "url": url_to_use,
                "cached_files": cached_files_info if cached_files_info else None,
                "original_identifier": original_identifier, "is_heartbeat_type": is_heartbeat_type,
                "timestamp_failed": time.time(), "cache_retry_attempts": 0
            }
            
            self.failed_sends_cache.append(cache_item)

            if self._save_cache(): # _save_cache does not acquire lock
                log_msg = f"[INFO] Added failed send ({original_identifier}) to cache."
                if cached_files_info: log_msg += f" {len(cached_files_info)} file(s) associated."
                log_msg += f" Cache size: {len(self.failed_sends_cache)}"
                if self.print_response or self.debug: print(log_msg)
                # _enforce_cache_limits will acquire the RLock again, which is fine.
                self._enforce_cache_limits(save_after_enforce=True)
            else:
                self.failed_sends_cache.pop()
                for path in temp_files_written_paths:
                    try:
                        if os.path.exists(path): os.remove(path)
                    except OSError as e_del:
                        if self.print_response or self.debug: print(f"[ERROR] Failed to clean up temp file {path} after cache save failure: {e_del}")
                
                err_msg = (f"[CRITICAL] Failed to save cache after attempting to add '{original_identifier}'. "
                           "Item and its temporary files have been rolled back from this caching attempt.")
                messages_list_for_warning.append(err_msg)
                if self.print_response or self.debug: print(err_msg)

    def _remove_from_cache_by_item(self, item_to_remove):
        with self.cache_lock: # Acquire lock
            self._remove_files_for_cached_item(item_to_remove) # This helper does not lock
            try:
                self.failed_sends_cache.remove(item_to_remove)
                if not self._save_cache(): # _save_cache does not lock
                    if self.print_response or self.debug:
                        print(f"[CRITICAL] Failed to save cache after removing item '{item_to_remove.get('original_identifier')}' from memory. "
                              "Disk cache may be inconsistent until next successful save or restart.")
                if self.print_response or self.debug:
                    print(f"[INFO] Removed item ({item_to_remove.get('original_identifier')}) from cache. Cache size: {len(self.failed_sends_cache)}")
            except ValueError:
                if self.print_response or self.debug:
                    print(f"[WARNING] Item ({item_to_remove.get('original_identifier')}) not found in cache for removal (possibly already removed).")

    def _send_data_thread(self, data_payload, url_to_use, files_for_request, messages, identifier,
                          is_heartbeat_type, cache_entry=None, files_opened_for_this_send=None):
        start_time = time.time()
        retries = 0
        single_request_timeout = max(5, self.timeout / (self.max_retries +1) if self.max_retries > 0 else self.timeout)


        try:
            while retries <= self.max_retries and (time.time() - start_time) < self.timeout: # Use <= for max_retries
                try:
                    current_files_for_request = {}
                    if files_for_request:
                        for field, (name, content_spec, mimetype) in files_for_request.items():
                            if isinstance(content_spec, io.IOBase): # Handles BytesIO and opened file objects
                                content_spec.seek(0)
                                current_files_for_request[field] = (name, content_spec, mimetype)
                            elif isinstance(content_spec, bytes):
                                current_files_for_request[field] = (name, content_spec, mimetype)

                    response = requests.post(url_to_use, headers=self.headers, data=data_payload,
                                             files=current_files_for_request if current_files_for_request else None,
                                             timeout=single_request_timeout)
                    response.raise_for_status()
                    messages.append(f"Data sent successfully to {url_to_use}: {response.status_code}")
                    if cache_entry:
                        # _remove_from_cache_by_item will acquire the lock
                        self._remove_from_cache_by_item(cache_entry)
                    return messages
                except requests.exceptions.Timeout:
                    messages.append(f"Request timeout (attempt {retries + 1}/{self.max_retries + 1}) to {url_to_use}")
                except requests.exceptions.RequestException as e:
                    messages.append(f"Error sending data (attempt {retries + 1}/{self.max_retries + 1}) to {url_to_use}: {e}")
                    if hasattr(e, 'response') and e.response is not None:
                        messages.append(f"Response status: {e.response.status_code}, Response text: {e.response.text[:200]}...")

                if self.debug and 'e' in locals() and hasattr(e, 'request'):
                    debug_details = [f"  [DEBUG] Request: {e.request.method} {e.request.url}"]
                    if e.request.headers: debug_details.append(f"  [DEBUG] Request Headers: {e.request.headers}")
                    if e.request.body:
                        body_repr = repr(e.request.body)
                        debug_details.append(f"  [DEBUG] Request Body: {body_repr[:200] + '...' if len(body_repr) > 200 else body_repr}")
                    if hasattr(e, 'response') and e.response is not None:
                        debug_details.append(f"  [DEBUG] Response Status: {e.response.status_code}")
                        if e.response.headers: debug_details.append(f"  [DEBUG] Response Headers: {e.response.headers}")
                    messages.extend(debug_details)
                
                if 'e' in locals():
                    del e # Clear e to avoid using stale exception info if next iteration doesn't raise

                retries += 1
                if retries <= self.max_retries and (time.time() - start_time) < self.timeout :
                    sleep_time = self.retry_delay * (2**(retries-1)) + random.uniform(0, 1) # Adjusted exponent
                    sleep_time = min(sleep_time, 30) # Cap sleep time
                    messages.append(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    break # Exit loop if max_retries reached or timeout exceeded

            messages.append(f"Failed to send data for '{identifier}' after {retries} attempts or overall timeout.")
            if not cache_entry and not is_heartbeat_type:
                if url_to_use and self.cache_files_dir is not None:
                    # _add_to_cache will acquire the lock
                    self._add_to_cache(data_payload, url_to_use, files_for_request, identifier, is_heartbeat_type, messages)
                elif not url_to_use:
                    messages.append(f"URL for '{identifier}' was invalid, not caching.")
                elif self.cache_files_dir is None:
                    messages.append(f"Cache files directory not available for '{identifier}', not caching files.")

            return messages
        finally:
            if files_opened_for_this_send:
                if self.debug: messages.append(f"[DEBUG] Closing {len(files_opened_for_this_send)} file objects opened for this send.")
                for f_obj in files_opened_for_this_send:
                    try:
                        f_obj.close()
                    except Exception as e_close:
                        messages.append(f"[WARNING] Error closing cached file object for '{identifier}': {e_close}")

    def _thread_done_callback(self, future, identifier, data_type_description):
        if not self.print_response: return
        try:
            messages = future.result()
            is_successful = any("Data sent successfully" in msg for msg in messages)
            prefix = "[SUCCESS]" if is_successful else "[FAILURE]"
            # Consider a dedicated print lock if output interleaving is a major issue
            # with self.print_lock:
            print(f"\n{prefix} {data_type_description}: {identifier}")
            for msg in messages: print(f"  - {msg}")
            print("-" * (len(identifier) + len(data_type_description) + len(prefix) + 5))
        except Exception as e:
            # with self.print_lock:
            print(f"\n[ERROR] Thread execution error for {data_type_description}: {identifier}")
            print(f"  - An error occurred in the thread: {e}")
            if self.debug:
                import traceback
                print(f"  - Traceback: {traceback.format_exc()}")
            print("-" * (len(identifier) + len(data_type_description) + 25))


    def send_data(self, data, heartbeat=False, files=None):
        messages = []
        base_id = "Heartbeat" if heartbeat else "DataUpload"
        # Using uuid for more unique identifier if multiple sends happen in same fraction of second
        identifier = f"{base_id} - {uuid.uuid4()}"


        url_to_use = self.heartbeat_url if heartbeat else self.api_url
        if not url_to_use:
            err_msg = f"URL not configured for {'heartbeat' if heartbeat else 'data upload'}. Cannot send: {identifier}"
            if self.print_response or self.debug: print(f"[ERROR] {err_msg}")
            return

        data_type_desc = "Heartbeat" if heartbeat else "Data Upload"
        files_to_send_prepared = None
        if files:
            data_type_desc += " with Files"
            files_to_send_prepared = {}
            for field, (name, content, mimetype) in files.items():
                if isinstance(content, (io.BytesIO, bytes)):
                    files_to_send_prepared[field] = (name, content, mimetype)
                else:
                    msg = f"[WARNING] Unsupported file content type for field '{field}' in '{identifier}'. Expected BytesIO or bytes. Skipping this file."
                    messages.append(msg)
                    if self.print_response or self.debug: print(msg)
            if not files_to_send_prepared: files_to_send_prepared = None

        future = self.executor.submit(
            self._send_data_thread,
            data, url_to_use, files_to_send_prepared,
            messages, identifier, heartbeat,
            cache_entry=None,
            files_opened_for_this_send=None
        )
        future.add_done_callback(lambda f: self._thread_done_callback(f, identifier, data_type_desc))

    def _start_cache_management_timer(self):
        if self.shutting_down_event.is_set() or self.cache_retry_interval <= 0:
            return

        self.cache_management_timer = threading.Timer(self.cache_retry_interval, self._periodic_cache_management_task)
        self.cache_management_timer.daemon = True
        self.cache_management_timer.start()
        if self.debug: print(f"[DEBUG] Cache management timer scheduled to run in {self.cache_retry_interval}s.")

    def _periodic_cache_management_task(self):
        if self.shutting_down_event.is_set():
            if self.debug: print("[DEBUG] Shutdown signalled, skipping periodic cache management.")
            return

        if self.debug: print("[DEBUG] Running periodic cache management (enforcing limits and retrying)...")
        # Both these methods will acquire self.cache_lock (RLock) as needed.
        self._enforce_cache_limits(save_after_enforce=True)
        self._retry_failed_sends()

        if not self.shutting_down_event.is_set():
            self._start_cache_management_timer()
        else:
            if self.debug: print("[DEBUG] Shutdown signalled after periodic cache management, not rescheduling.")

    def _enforce_cache_limits(self, save_after_enforce=False):
        with self.cache_lock: # Acquire lock
            changed_in_memory = False
            current_time = time.time()

            if self.max_cache_age_seconds > 0:
                initial_count = len(self.failed_sends_cache)
                # Iterate backwards for safe removal while iterating
                original_cache = list(self.failed_sends_cache) # Iterate over a copy
                new_cache = []
                items_removed_by_age = []

                for item in original_cache:
                    if current_time - item.get('timestamp_failed', current_time) > self.max_cache_age_seconds:
                        if self.print_response or self.debug:
                            print(f"[INFO] Pruning aged cache item (older than {self.max_cache_age_seconds}s): {item['original_identifier']}")
                        self._remove_files_for_cached_item(item) # Does not lock
                        items_removed_by_age.append(item)
                        changed_in_memory = True
                    else:
                        new_cache.append(item)
                self.failed_sends_cache = new_cache
                
                if self.debug and changed_in_memory: # items_removed_by_age:
                    print(f"[DEBUG] Pruned {len(items_removed_by_age)} items by age.")


            if self.max_cache_items > 0 and len(self.failed_sends_cache) > self.max_cache_items:
                num_to_prune = len(self.failed_sends_cache) - self.max_cache_items
                if self.print_response or self.debug:
                    print(f"[INFO] Cache size ({len(self.failed_sends_cache)}) exceeds limit ({self.max_cache_items}). Pruning {num_to_prune} oldest items.")
                
                items_pruned_by_count = self.failed_sends_cache[:num_to_prune]
                self.failed_sends_cache = self.failed_sends_cache[num_to_prune:]
                for item in items_pruned_by_count:
                    self._remove_files_for_cached_item(item) # Does not lock
                changed_in_memory = True
                if self.debug: print(f"[DEBUG] Pruned {num_to_prune} items by count.")

            if changed_in_memory and save_after_enforce:
                if not self._save_cache(): # Does not lock
                     if self.print_response or self.debug:
                        print(f"[CRITICAL] Failed to save cache after enforcing limits. Disk cache may be inconsistent.")
            elif self.debug and not changed_in_memory:
                print("[DEBUG] Cache limits checked, no changes needed.")


    def _cleanup_orphaned_cached_files(self):
        if not self.cache_files_dir or not os.path.isdir(self.cache_files_dir):
            if self.debug: print("[DEBUG] Cache files directory not configured or does not exist, skipping orphan cleanup.")
            return

        if self.print_response or self.debug:
            print(f"[INFO] Starting cleanup of orphaned files in {self.cache_files_dir}...")

        valid_cached_files = set()
        with self.cache_lock: # Acquire lock to safely read failed_sends_cache
            for item in self.failed_sends_cache:
                if item.get("cached_files"):
                    for _, file_info in item["cached_files"].items():
                        if file_info.get("cached_filepath"):
                            valid_cached_files.add(os.path.normpath(file_info["cached_filepath"]))

        cleaned_count = 0
        try:
            for filename in os.listdir(self.cache_files_dir):
                filepath = os.path.normpath(os.path.join(self.cache_files_dir, filename))
                if os.path.isfile(filepath):
                    if filepath not in valid_cached_files:
                        try:
                            os.remove(filepath)
                            cleaned_count += 1
                            if self.debug: print(f"  - Removed orphaned cache file: {filepath}")
                        except OSError as e:
                            if self.print_response or self.debug:
                                print(f"  - [ERROR] Failed to remove orphaned cache file {filepath}: {e}")
        except OSError as e:
            if self.print_response or self.debug:
                print(f"[ERROR] Could not list cache files directory {self.cache_files_dir} for cleanup: {e}")
            return

        if cleaned_count > 0 and (self.print_response or self.debug):
            print(f"[INFO] Orphan cleanup complete. Removed {cleaned_count} file(s).")
        elif self.debug: # Print only if debug and no other print happened
             if cleaned_count == 0 and not (self.print_response and cleaned_count > 0):
                print("[DEBUG] Orphan cleanup complete. No orphaned files found.")


    def _retry_failed_sends(self):
        items_to_process_this_round = []
        with self.cache_lock: # Acquire lock for initial scan
            if not self.failed_sends_cache:
                if self.debug: print("[DEBUG] Cache is empty, no items to retry.")
                return

            if self.print_response or self.debug:
                print(f"[INFO] Checking cache for items to retry. Current cache size: {len(self.failed_sends_cache)}")

            for item in self.failed_sends_cache:
                if item.get('cache_retry_attempts', 0) < self.max_cache_retries:
                    items_to_process_this_round.append(item) # Appending a reference, not a deep copy
                elif self.print_response or self.debug:
                     print(f"[INFO] Max cache retries ({self.max_cache_retries}) already met for item: {item['original_identifier']}. It will be pruned by age/size limits if applicable.")

        processed_count = 0
        # Iterate over a list of references. The actual items might be modified or removed from self.failed_sends_cache
        # by other operations or by this loop itself.
        for item_snapshot_ref in items_to_process_this_round: # item_snapshot_ref is a reference to an item in cache
            if self.shutting_down_event.is_set():
                if self.debug: print("[DEBUG] Shutdown signalled, stopping cache retry cycle.")
                break

            if processed_count >= self.max_workers:
                if self.print_response or self.debug:
                    print(f"[INFO] Pausing cache retry cycle after processing {processed_count} items to not overwhelm workers.")
                break
            
            live_item_for_retry = None
            with self.cache_lock: # Acquire lock for updating/removing item
                # Verify item still exists in cache and is the one we expect, then update it
                try:
                    # Find the item in the current cache list. It might have been removed.
                    # We use the reference itself to find it, assuming it's still valid.
                    if item_snapshot_ref not in self.failed_sends_cache:
                        if self.debug: print(f"[DEBUG] Item {item_snapshot_ref.get('original_identifier')} no longer in live cache, skipping retry.")
                        continue
                    
                    # Now item_snapshot_ref is confirmed to be a 'live' item from the cache
                    live_item = item_snapshot_ref 
                    
                    current_attempts = live_item.get('cache_retry_attempts', 0)
                    if current_attempts >= self.max_cache_retries:
                        # This check is slightly redundant due to the initial filtering,
                        # but good for safety if state changed between snapshot and now.
                        if self.print_response or self.debug:
                            print(f"[INFO] Max cache retries ({self.max_cache_retries}) reached for item: {live_item['original_identifier']}. Removing from cache.")
                        # _remove_from_cache_by_item will re-acquire RLock, which is fine.
                        self._remove_from_cache_by_item(live_item)
                        continue

                    live_item['cache_retry_attempts'] = current_attempts + 1
                    
                    if not self._save_cache(): # Does not lock
                        if self.print_response or self.debug:
                            print(f"[CRITICAL] Failed to save cache after updating retry count for {live_item['original_identifier']}. Proceeding with retry attempt.")
                    
                    live_item_for_retry = live_item # This is the item to use for submission

                except Exception as e_lock:
                    if self.print_response or self.debug:
                        print(f"[ERROR] Unexpected error during cache lock for retry prep of {item_snapshot_ref.get('original_identifier')}: {e_lock}")
                    continue
            
            # If live_item_for_retry was successfully prepared, submit it
            if live_item_for_retry:
                messages = []
                retry_identifier = f"CachedRetry - OrigID: {live_item_for_retry['original_identifier']} - Attempt: {live_item_for_retry['cache_retry_attempts']}"
                data_type_desc = "Cached Heartbeat Retry" if live_item_for_retry['is_heartbeat_type'] else "Cached Data Retry"

                files_for_request_retry = {}
                opened_retry_files_objects = []
                all_cached_files_opened_successfully = True

                if live_item_for_retry.get("cached_files"):
                    data_type_desc += " with Files"
                    if self.print_response or self.debug:
                        print(f"[INFO] Retrying {retry_identifier} with {len(live_item_for_retry['cached_files'])} file(s) from disk.")

                    for field_name, file_info in live_item_for_retry["cached_files"].items():
                        try:
                            f_obj = open(file_info["cached_filepath"], 'rb')
                            opened_retry_files_objects.append(f_obj)
                            files_for_request_retry[field_name] = (
                                file_info["original_filename"], f_obj, file_info["mimetype"]
                            )
                        except Exception as e_open:
                            messages.append(f"[ERROR] Failed to open cached file {file_info['cached_filepath']} for retry: {e_open}")
                            if self.print_response or self.debug:
                                print(f"[ERROR] Retrying {retry_identifier}: Could not open {file_info['cached_filepath']}: {e_open}")
                            all_cached_files_opened_successfully = False
                    if not all_cached_files_opened_successfully:
                        messages.append(f"[WARNING] Retrying {retry_identifier} but one or more cached files could not be opened.")
                else:
                    if self.print_response or self.debug:
                        print(f"[INFO] Retrying from cache (no files associated): {retry_identifier}")

                future = self.executor.submit(
                    self._send_data_thread,
                    live_item_for_retry['data_payload'], live_item_for_retry['url'],
                    files_for_request_retry if files_for_request_retry else None,
                    messages, retry_identifier, live_item_for_retry['is_heartbeat_type'],
                    cache_entry=live_item_for_retry, # Pass the live cache_entry
                    files_opened_for_this_send=opened_retry_files_objects
                )
                future.add_done_callback(lambda f: self._thread_done_callback(f, retry_identifier, data_type_desc))
                processed_count += 1
                time.sleep(0.1)

        if processed_count > 0 and (self.print_response or self.debug):
            print(f"[INFO] Cache retry cycle submitted {processed_count} items for processing.")
        elif not items_to_process_this_round and (self.print_response or self.debug) and self.failed_sends_cache:
            # This condition might be tricky if items were removed during processing
            if self.debug: print(f"[DEBUG] No cache items initially eligible or remaining for retry at this time.")


    def send_heartbeat(self, sn, timestamp, status_log="Heartbeat received successfully."):
        if not self.heartbeat_url:
            if self.print_response or self.debug: print("[ERROR] Heartbeat URL not configured. Cannot send heartbeat.")
            return
        data = {
            "sn": sn, "mac_address": self.mac_address, "ip_address": self.ip_address,
            "hw_platform": "opi", "host_name": f"{self.os_username}@{self.hostname}",
            "status_log": status_log, "time": timestamp
        }
        if self.print_response or self.debug:
            print(f"[INFO] Preparing heartbeat: SN={sn}, Time={timestamp}")
            if self.debug: print(f"[DEBUG] Heartbeat data: {data}")
        self.send_data(data, heartbeat=True)

    def shutdown(self, wait=True):
        if self.print_response or self.debug: print("[INFO] Shutting down DataUploader...")
        self.shutting_down_event.set()

        if self.cache_management_timer:
            if self.print_response or self.debug: print("[INFO] Cancelling cache management timer...")
            self.cache_management_timer.cancel()
            # Attempt to join, but with a timeout as it's a daemon thread
            # and might be in a sleep state. The shutting_down_event should help it exit.
            # self.cache_management_timer.join(timeout=self.cache_retry_interval + 2)


        if self.print_response or self.debug: print("[INFO] Shutting down DataUploader executor...")
        self.executor.shutdown(wait=wait)

        if self.print_response or self.debug: print("[INFO] Performing final cache save...")
        with self.cache_lock: # Acquire lock for final save
            self._save_cache() # Does not lock

        if self.print_response or self.debug: print("[INFO] DataUploader shut down complete.")