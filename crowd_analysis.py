import time
import cv2
import numpy as np
import tensorflow as tf
import json
import paho.mqtt.client as mqtt
import logging
import os

# Disable oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CrowdAnalyzer")

class CrowdAnalyzer:
    def __init__(self, model_path="models/ssd_mobilenet.tflite", edge_mode=False):
        """Initialize crowd analyzer with edge mode support"""
        self.edge_mode = edge_mode
        
        # Edge configurations
        self.MQTT_BROKER = "192.168.1.100"  # Your server IP
        self.MQTT_TOPIC = "crowd/edge_updates"
        self.MAX_PEOPLE_THRESHOLD = 10  # Trigger alert if exceeded
        self.HEARTBEAT_INTERVAL = 30  # Seconds between normal updates
        self.last_send_time = 0
        
        # Initialize MQTT if in edge mode
        self.mqtt_client = None
        if self.edge_mode:
            self.setup_mqtt()

        # Load model
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]['shape'][1:3]  # (height, width)
            logger.info(f"Model loaded. Input shape: {self.input_shape}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Data tracking
        self.people_counts = []
        self.anomalies_log = []
        self.processing_times = []
        self._frame_counter = 0

    def setup_mqtt(self):
        """Initialize MQTT client for edge devices"""
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
        try:
            self.mqtt_client.connect(self.MQTT_BROKER, 1883, 60)
            self.mqtt_client.loop_start()
            logger.info("MQTT client connected")
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")

    def on_mqtt_connect(self, client, userdata, flags, rc):
        logger.info(f"MQTT connected with result code {rc}")

    def on_mqtt_disconnect(self, client, userdata, rc):
        logger.warning(f"MQTT disconnected with result code {rc}")

    def process_frame(self, frame):
        """Process frame and detect people with bounding boxes"""
        start_time = time.time()  # Define start_time at beginning
        
        try:
            # Frame skip logic for edge devices
            self._frame_counter += 1
            if self.edge_mode and self._frame_counter % 3 != 0:
                return frame, 0, [], []

            # Preprocess frame
            resized = cv2.resize(frame, (self.input_shape[1], self.input_shape[0]))
            input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)

            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()

            # Get results
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
            num_detections = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])

            # Process detections
            count = 0
            detections = []
            anomalies = []
            h, w = frame.shape[:2]

            for i in range(num_detections):
                if scores[i] > 0.5 and classes[i] == 0:  # Class 0 = person
                    ymin, xmin, ymax, xmax = boxes[i]
                    x1, y1 = int(xmin * w), int(ymin * h)
                    x2, y2 = int(xmax * w), int(ymax * h)
                    detections.append((x1, y1, x2, y2))
                    count += 1

                    # Mark as anomaly if count exceeds threshold
                    if count > self.MAX_PEOPLE_THRESHOLD:
                        anomalies.append((x1, y1, x2, y2))

            # Store metrics
            processing_time = (time.time() - start_time) * 1000
            self.people_counts.append(count)
            self.anomalies_log.append(len(anomalies))
            self.processing_times.append(round(processing_time, 2))

            # Send data if in edge mode
            if self.edge_mode:
                self.send_edge_data(count, detections, anomalies)

            logger.debug(f"Detected {count} people")
            return frame, count, detections, anomalies

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame, 0, [], []

    def send_edge_data(self, count, detections, anomalies):
        """Send data via MQTT for edge devices"""
        current_time = time.time()
        
        # Only send if anomalies or threshold exceeded or heartbeat interval passed
        if (count <= self.MAX_PEOPLE_THRESHOLD and 
            not anomalies and 
            current_time - self.last_send_time < self.HEARTBEAT_INTERVAL):
            return

        payload = {
            "count": count,
            "anomalies": len(anomalies),
            "timestamp": current_time,
            "detections": detections
        }

        if self.mqtt_client and self.mqtt_client.is_connected():
            try:
                self.mqtt_client.publish(
                    self.MQTT_TOPIC,
                    payload=json.dumps(payload),
                    qos=1
                )
                self.last_send_time = current_time
                logger.info(f"Data sent via MQTT: People={count}, Anomalies={len(anomalies)}")
            except Exception as e:
                logger.error(f"MQTT publish failed: {e}")

    def cleanup(self):
        """Clean up resources"""
        if self.mqtt_client and self.mqtt_client.is_connected():
            self.mqtt_client.disconnect()
            self.mqtt_client.loop_stop()