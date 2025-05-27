import csv
import requests
import time
import os
import socket
import fcntl
import struct
import threading
import random
from concurrent.futures import ThreadPoolExecutor
import getpass

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
        self.heartbeat_url = heartbeat_url
        self.headers = headers or {"Content-Type": "application/json"}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.print_response = print_response
        self.mac_address = self._get_mac_address()
        self.ip_address = self._get_ip_address()
        self.hostname = os.uname().nodename
        try:
            self.os_username = os.getlogin()
        except OSError:
            # Fallback for environments where os.getlogin() fails (e.g., some daemons, cron, or Docker)
            self.os_username = getpass.getuser()

    def _get_ip_address(self):
        """
        Retrieves the IP address of 'eth0' if connected, otherwise 'wlan0' if connected.
        Returns:
            str: The IP address in the format 'X.X.X.X', or '127.0.0.1' if network not connected.
        """
        def get_ip(ifname):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                return socket.inet_ntoa(fcntl.ioctl(
                    s.fileno(),
                    0x8915,  # SIOCGIFADDR
                    struct.pack('256s', ifname[:15].encode('utf-8'))
                )[20:24])
            except OSError:
                return None

        for iface in ['eth0', 'wlan0']:
            # Check if interface exists and is up
            if os.path.exists(f'/sys/class/net/{iface}/operstate'):
                with open(f'/sys/class/net/{iface}/operstate') as f:
                    state = f.read().strip()
                if state == 'up':
                    ip = get_ip(iface)
                    if ip and ip != '127.0.0.1':
                        return ip
        return "127.0.0.1"

    def _get_mac_address(self):
        """        Retrieves the MAC address of the first network interface.
        Returns:        
            str: The MAC address in the format 'XX:XX:XX:XX:XX:XX'.
        """
        try:
            with open('/sys/class/net/eth0/address', 'r') as f:
                mac_address = f.read().strip()
            return mac_address
        except FileNotFoundError:
            print("Fallback to using the first network interface")
            import uuid

            mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(0,2*6,2)][::-1])
            return mac
    
    # Add image resize function

    def _send_data_thread(self, data, heartbeat, files, messages, identifier):
        """
        Sends data to the server with retry and backoff.
        """
        start_time = time.time()
        retries = 0

        while retries < self.max_retries and time.time() - start_time < self.timeout:
            try:
                url = self.heartbeat_url if heartbeat else self.api_url
                # if files:
                #     # If files are provided, send data as form-data
                #     response = requests.post(url, headers=self.headers, data=data, files=files)
                # else:
                #     # Otherwise, send data as JSON
                #     response = requests.post(url, headers=self.headers, json=data)
                # print(f"in _send_data_thread, url: {url}, data: {data}, files: {files}")
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

    def _thread_done_callback(self, future, identifier, heartbeat):
        if not self.print_response:
            return

        data_type = "Heartbeat" if heartbeat else "DataSend"

        try:
            messages = future.result()
            print(f"----- Thread Results for: {identifier} ({data_type}) -----")
            for msg in messages:
                print(msg)
            print("------------------------------------------\n")
        except Exception as e:
            print(f"----- Thread Error for: {identifier} ({data_type}) -----")
            print(f"An error occurred in the thread: {e}")
            print("------------------------------------------\n")

    def send_data(self, data, heartbeat=False, files=None):
        """
        Sends data to the server using the thread pool.
        """
        messages = []
        identifier = f"Data Upload - {time.time()}"  # Unique identifier for this upload attempt
        future = self.executor.submit(self._send_data_thread, data, heartbeat, files, messages, identifier)
        future.add_done_callback(lambda f: self._thread_done_callback(f, identifier, heartbeat))  # Add the callback

    def send_heartbeat(self, sn, ip_address, time, status_log="Heartbeat received successfully."):
        """
        Sends a heartbeat signal to the server.
        """
        data = {
            "sn": sn,
            "mac_address": self.mac_address,
            "ip_address": ip_address if "127.0.0.1" in ip_address else self.ip_address,
            "hw_platform": "rpi",
            "host_name": self.os_username+"@"+self.hostname,
            "status_log": status_log,
            "time": time
        }

        print(data)
        self.send_data(data, heartbeat=True)