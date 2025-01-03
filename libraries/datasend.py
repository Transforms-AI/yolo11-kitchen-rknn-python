import csv
import requests
import time
import os

class DataUploader:
    def __init__(self, api_url, headers=None):
        """
        Initializes the DataUploader with the API endpoint URL and optional headers.

        Args:
            api_url: The URL of the API endpoint for uploading data.
            headers: Optional dictionary of headers to include in the request.
        """
        self.api_url = api_url
        self.headers = headers or {}

    def send_data(self, data, files=None):
        """
        Sends data to the server.

        Args:
            data: A dictionary containing the data to be sent in the request body.
            files: Optional dictionary of files to be uploaded. The keys should be the
                   names of the form fields for the files, and the values should be
                   file-like objects or tuples of (filename, file-like object).

        Returns:
            A list of messages indicating the success or failure of the data upload.
        """
        messages = []

        try:
            response = requests.post(self.api_url, headers=self.headers, data=data, files=files)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            messages.append(f"Data sent successfully: {response.status_code}")

        except requests.exceptions.RequestException as e:
            messages.append(f"Error sending data: {e}")
            if 'response' in locals() and response:  # Check if response exists and is not None
                messages.append(f"Response text: {response.text}")
           

        return messages