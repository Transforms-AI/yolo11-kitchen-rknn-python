import csv
import requests
import time
import os
import threading

class DataUploader:
    def __init__(self, api_url, heartbeat_url=None, headers=None):
        """
        Initializes the DataUploader with the API endpoint URL and optional headers.

        Args:
            api_url: The URL of the API endpoint for uploading data.
            heartbeat_url: The URL of the heartbeat endpoint for uploading data.
            headers: Optional dictionary of headers to include in the request.
        """
        self.api_url = api_url
        self.hearbeat_url = heartbeat_url
        self.headers = headers or {}
        self.lock = threading.Lock()  # Create a lock for thread safety

    def _send_data_thread(self, data, heartbeat, files, messages):
        """
        Sends data to the server in a separate thread.

        Args:
            data: A dictionary containing the data to be sent in the request body.
            heartbeat: A boolean indicating whether this is a heartbeat request.
            files: Optional dictionary of files to be uploaded.
            messages: A list to store messages from the thread.
        """
        try:
            url = self.hearbeat_url if heartbeat else self.api_url
            response = requests.post(url, headers=self.headers, data=data, files=files)
            response.raise_for_status()

            with self.lock:  # Acquire lock to safely modify the shared list
                messages.append(f"Data sent successfully: {response.status_code}")

        except requests.exceptions.RequestException as e:
            with self.lock:
                messages.append(f"Error sending data: {e}")
                if 'response' in locals() and response:
                    messages.append(f"Response text: {response.text}")

    def send_data(self, data, heartbeat=False, files=None):
        """
        Sends data to the server using a thread.

        Args:
            data: A dictionary containing the data to be sent in the request body.
            heartbeat: A boolean indicating whether this is a heartbeat request.
            files: Optional dictionary of files to be uploaded. The keys should be the
                   names of the form fields for the files, and the values should be
                   file-like objects or tuples of (filename, file-like object).

        Returns:
            A list of messages indicating the success or failure of the data upload.
        """
        messages = []
        thread = threading.Thread(target=self._send_data_thread, args=(data, heartbeat, files, messages))
        thread.start()
        thread.join()  # Wait for the thread to complete
        return messages

    def send_heartbeat(self, sn, ip, time):
        """
        Sends a heartbeat signal to the server.

        Args:
            sn: The serial number of the device.
            ip: The IP address of the device.
            time: The current timestamp.

        Returns:
            A list of messages indicating the success or failure of the heartbeat upload.
        """
        data = {
            "sn": sn,
            "version": 2,  # Replace with actual version if available
            "ip_address": ip,
            "time_zone": 3,  # Replace with actual time zone if available
            "hw_platform": "Platform_XYZ",  # Replace with actual hardware platform if available
            "time": time
        }

        return self.send_data(data, heartbeat=True)