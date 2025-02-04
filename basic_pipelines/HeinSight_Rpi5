import os
import sys
import time
import queue
import cv2
import numpy as np
import datetime
import threading
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import List
from PIL import Image
from itertools import chain
from random import randint
from utils import HailoAsyncInference  # Import Hailo inference
from object_detection_utils import ObjectDetectionUtils

class HeinSightHailo:
    def __init__(self, vial_model_path, contents_model_path):  # Add parameters here
        """
        Initialize HeinSight with Hailo models.
        """
        self._running = True
        
        # Create input and output queues
        self.vial_input_queue = queue.Queue()
        self.vial_output_queue = queue.Queue()
        self.contents_input_queue = queue.Queue()
        self.contents_output_queue = queue.Queue()

        # Initialize Hailo models correctly
        self.vial_model = HailoAsyncInference(vial_model_path, self.vial_input_queue, self.vial_output_queue, batch_size=1)
        self.contents_model = HailoAsyncInference(contents_model_path, self.contents_input_queue, self.contents_output_queue, batch_size=1)

        self.vial_location = None
        self.vial_size = [80, 200]
        self.color_palette = self._register_colors(["Homo", "Hetero", "Solid", "Residue", "Empty"])
        self.output_dataframe = pd.DataFrame()
        self.output_frame = None

    def preprocess_for_hailo(self, frame):
        """
        Convert OpenCV BGR frame to Hailo-compatible input format.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    def extract_hailo_detections(self, infer_results):
        """
        Convert Hailo inference output to bounding boxes and labels.
        """
        detections = []
        for result in infer_results:
            bbox = result["bbox"]  # Assuming Hailo outputs a dictionary
            label = result["label"]
            confidence = result["confidence"]
            detections.append([*bbox, confidence, label])
        return np.array(detections)

    def find_vial(self, frame):
        """
        Detect the vial in the video frame using Hailo.
        """
        processed_frame = self.preprocess_for_hailo(frame)
        result = self.vial_model.run(processed_frame)
        bboxes = self.extract_hailo_detections(result)

        if len(bboxes) == 0:
            return None
        self.vial_location = [int(x) for x in bboxes[0][:4]]
        self.vial_size = [
            self.vial_location[2] - self.vial_location[0],
            int((self.vial_location[3] - self.vial_location[1]) * (1 - self.CAP_RATIO))
        ]
        return bboxes

    def content_detection(self, vial_frame):
        """
        Detect content inside the vial using Hailo.
        """
        processed_frame = self.preprocess_for_hailo(vial_frame)
        result = self.contents_model.run(processed_frame)
        return self.extract_hailo_detections(result)

    def process_vial_frame(self, vial_frame, update_od=True):
        """
        Process a single frame, detect content, and overlay results.
        """
        if update_od:
            bboxes = self.content_detection(vial_frame)
        else:
            bboxes = []

        if self.INCLUDE_BB:
            frame = self.draw_bounding_boxes(vial_frame, bboxes)
        return frame

    def draw_bounding_boxes(self, image, bboxes):
        """
        Draw bounding boxes on the image.
        """
        output_image = image.copy()
        for rect in bboxes:
            x1, y1, x2, y2 = [int(x) for x in rect[:4]]
            label = rect[-1]
            color = self.color_palette.get(label, (255, 255, 255))
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return output_image

    def run(self, source=0, output_name="output", fps=5, res=(1920, 1080)):
        """
        Main function for real-time monitoring.
        """
        logger.info("Starting HeinSight with Hailo on live camera.")

        video = cv2.VideoCapture(source)
        video.set(cv2.CAP_PROP_FPS, fps)
        video.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])

        while self._running:
            ret, frame = video.read()
            if not ret:
                break

            # Detect vial on the first frame
            if self.vial_location is None:
                result = self.find_vial(frame)
                if result is None:
                    logger.warning("No vial detected. Retrying...")
                    continue

            # Crop and process vial region
            vial_frame = frame[self.vial_location[1]:self.vial_location[3], self.vial_location[0]:self.vial_location[2]]
            frame_image = self.process_vial_frame(vial_frame, update_od=True)

            # Display real-time output
            if self.VISUALIZE:
                cv2.imshow("HeinSight Hailo", frame_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        video.release()
        cv2.destroyAllWindows()
        logger.info(f"Results saved as {output_name}.mkv")

    @staticmethod
    def _register_colors(class_names):
        """
        Assign unique colors for detected classes.
        """
        name_color_dict = {}
        for name in class_names:
            name_color_dict[name] = (randint(0, 255), randint(0, 255), randint(0, 255))
        return name_color_dict


if __name__ == "__main__":
    heinsight = HeinSightHailo(vial_model_path="models/vial_detection.hef", contents_model_path="models/contents_detection.hef")
    heinsight.run(0)  # Start with USB camera
