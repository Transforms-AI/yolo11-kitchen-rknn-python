import cv2
import threading
import paho.mqtt.client as mqtt
import time
import base64

class StreamPublisher:
    
    def __init__(self,topic, start_stream=True, host="127.0.0.1", port=1883 ) -> None :
        """
        Construct a new 'stream_publisher' object to broadcast a video stream using Mosquitto_MQTT

        :param topic: MQTT topic to send Stream 
        :param video_address: link for OpenCV to read stream from, default 0 (webcam)

        :param start_stream:  start streaming while making object, default True, else call object.start_streaming()
        
        :param host:  IP address of Mosquitto MQTT Broker
        :param Port:  Port at which Mosquitto MQTT Broker is listening
        
        :return: returns nothing
        """
        
        self.client = mqtt.Client()  # create new instance
        
        print(f"host {host}")
        self.client.connect(host, port)
        self.topic = topic
        self.img = None
        self.stop_stream = False
        self.start_stream = start_stream
        self.streaming_thread = None
        #self.video_source=video_address
        self.read_lock = threading.Lock()
        
        if self.start_stream:
            self.start_streaming()
    
    def start_streaming(self):
        self.start_stream = True
        self.stop_stream = False
        self.streaming_thread= threading.Thread(target=self.stream)
        self.streaming_thread.start()

    def stop_streaming(self):
        self.stop_stream = True
        self.start_stream = False
        if self.streaming_thread is not None:
            self.streaming_thread.join()


    def updateFrame(self,frame):
        with self.read_lock:
            self.img = frame.copy()   

    def liveStream(self, status):
        self.start_stream = status

    def stream(self):
        print("Streaming Started.....")
        while True:
            if self.img is not None and self.start_stream:
                with self.read_lock:
                    img = self.img
                    img = cv2.resize(img, (640 ,420))  # to reduce resolution 
                    img_str = cv2.imencode('.jpg', img,[cv2.IMWRITE_JPEG_QUALITY,70])[1]
                    image_base64 = base64.b64encode(img_str).decode('utf-8')
                    self.client.publish(self.topic, image_base64)
                    self.img = None
            elif self.stop_stream:
                print("Streaming Stopped.....")
                break
                
            time.sleep(0.03)
                