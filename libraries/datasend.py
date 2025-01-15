import csv
import requests
import time
import os
import threading
import random
from concurrent.futures import ThreadPoolExecutor

class DataUploader:
    def __init__(self, api_url, heartbeat_url=None, headers=None, max_workers=5, max_retries=5, retry_delay=1, timeout=10, print_response=True):
        """
        Initializes the DataUploader.

        Args:
            api_url: The URL of the API endpoint for uploading data.
            heartbeat_url: The URL of the heartbeat endpoint.
            headers: Optional dictionary of headers to include in the request.
            max_workers: The maximum number of threads in the thread pool.
            max_retries: The maximum number of retry attempts.
            retry_delay: The initial delay between retries (in seconds).
            timeout: The maximum total time for retries (in seconds).
            print_response: Whether to print the response and data from threads.
        """
        self.api_url = api_url
        self.hearbeat_url = heartbeat_url
        self.headers = headers or {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.print_response = print_response

    def _send_data_thread(self, data, heartbeat, files, messages, identifier):
        """
        Sends data to the server with retry and backoff.
        """
        start_time = time.time()
        retries = 0

        while retries < self.max_retries and time.time() - start_time < self.timeout:
            try:
                url = self.hearbeat_url if heartbeat else self.api_url
                response = requests.post(url, headers=self.headers, data=data, files=files)
                response.raise_for_status()

                with self.lock:
                    messages.append(f"Data sent successfully: {response.status_code}")
                return messages
            except requests.exceptions.RequestException as e:
                with self.lock:
                    messages.append(f"Error sending data (attempt {retries + 1}): {e}")
                    if 'response' in locals() and response:
                        messages.append(f"Response text: {response.text}")

                retries += 1
                if retries < self.max_retries:
                    sleep_time = self.retry_delay * (2**retries) + random.uniform(0, 1)
                    with self.lock:
                        messages.append(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)

        with self.lock:
            messages.append("Failed to send data after multiple retries.")
        return messages

    def _thread_done_callback(self, future, identifier):
        """
        Callback function executed when a thread finishes.
        """
        if not self.print_response:
            return

        try:
            messages = future.result()  # Get the result (or exception) from the thread
            print(f"----- Thread Results for: {identifier} -----")
            for msg in messages:
                print(msg)
            print("------------------------------------------\n")
        except Exception as e:
            print(f"----- Thread Error for: {identifier} -----")
            print(f"An error occurred in the thread: {e}")
            print("------------------------------------------\n")

    def send_data(self, data, heartbeat=False, files=None):
        """
        Sends data to the server using the thread pool.
        """
        messages = []
        identifier = f"Data Upload - {time.time()}"  # Unique identifier for this upload attempt
        future = self.executor.submit(self._send_data_thread, data, heartbeat, files, messages, identifier)
        future.add_done_callback(lambda f: self._thread_done_callback(f, data, identifier))  # Add the callback

    def send_heartbeat(self, sn, ip, time):
        """
        Sends a heartbeat signal to the server.
        """
        data = {
            "sn": sn,
            "version": 2,
            "ip_address": ip,
            "time_zone": 3,
            "hw_platform": "Platform_XYZ",
            "time": time
        }
        self.send_data(data, heartbeat=True)