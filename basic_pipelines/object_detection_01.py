#!/usr/bin/env python3

import argparse
import os
import sys
import queue
import threading
import cv2  # OpenCV for camera input
import numpy as np
from pathlib import Path
from PIL import Image
from loguru import logger
from typing import List
from object_detection_utils import ObjectDetectionUtils

# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference, load_input_images, validate_images, divide_list_to_batches


def parse_args() -> argparse.Namespace:
    """Initialize argument parser for the script."""
    parser = argparse.ArgumentParser(description="Detection Example")

    parser.add_argument("-n", "--net", default="yolov7.hef", help="Path for the network in HEF format.")
    parser.add_argument("-i", "--input", default="zidane.jpg", help="Path to the input image or folder.")
    parser.add_argument("-b", "--batch_size", default=1, type=int, help="Number of images in one batch")
    parser.add_argument("-l", "--labels", default="coco.txt", help="Path to labels file (default: coco2017).")
    
    # New argument to use live USB camera
    parser.add_argument("--camera", action="store_true", help="Use live USB camera instead of an image file.")

    args = parser.parse_args()

    # Validate file paths if not using a camera
    if not args.camera:
        # if not os.path.exists(args.net):
        #    raise FileNotFoundError(f"Network file not found: {args.net}")
        if not args.camera and not os.path.exists(args.net):
            raise FileNotFoundError(f"Network file not found: {args.net}")
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input path not found: {args.input}")
        if not os.path.exists(args.labels):
            raise FileNotFoundError(f"Labels file not found: {args.labels}")

    return args


def process_output(output_queue: queue.Queue, utils: ObjectDetectionUtils) -> None:
    """Process and visualize the output results."""
    while True:
        result = output_queue.get()
        if result is None:
            break  # Exit loop if sentinel value is received

        processed_image, infer_results = result
        detections = utils.extract_detections(infer_results)

        # If using live camera, show results in a window
        np_image = np.array(processed_image)
        frame = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        utils.visualize(detections, processed_image, image_id=0, output_path=None, width=frame.shape[1], height=frame.shape[0])
        
        cv2.imshow("Hailo Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    output_queue.task_done()


def infer_live_camera(net_path: str, labels_path: str) -> None:
    """Runs object detection using a live USB camera feed."""
    utils = ObjectDetectionUtils(labels_path)
    input_queue = queue.Queue()
    output_queue = queue.Queue()

    hailo_inference = HailoAsyncInference(net_path, input_queue, output_queue, batch_size=1)
    height, width, _ = hailo_inference.get_input_shape()

    cap = cv2.VideoCapture(0)  # Use the first connected USB camera
    if not cap.isOpened():
        logger.error("Failed to access the camera.")
        return

    process_thread = threading.Thread(target=process_output, args=(output_queue, utils))
    process_thread.start()

    logger.info("Press 'q' to exit the live detection.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to grab frame from camera.")
            break

        # Convert OpenCV frame (BGR) to PIL image (RGB)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        processed_image = utils.preprocess(pil_image, width, height)

        input_queue.put([processed_image])  # Send frame to inference queue

    cap.release()
    cv2.destroyAllWindows()
    output_queue.put(None)  # Signal process thread to exit
    process_thread.join()


def infer_images(images: List[Image.Image], net_path: str, labels_path: str, batch_size: int) -> None:
    """Runs object detection on a batch of images."""
    utils = ObjectDetectionUtils(labels_path)
    input_queue = queue.Queue()
    output_queue = queue.Queue()

    hailo_inference = HailoAsyncInference(net_path, input_queue, output_queue, batch_size)
    height, width, _ = hailo_inference.get_input_shape()

    enqueue_thread = threading.Thread(target=enqueue_images, args=(images, batch_size, input_queue, width, height, utils))
    process_thread = threading.Thread(target=process_output, args=(output_queue, utils))

    enqueue_thread.start()
    process_thread.start()
    hailo_inference.run()

    enqueue_thread.join()
    output_queue.put(None)  # Signal process thread to exit
    process_thread.join()


def main() -> None:
    """Main function to run the script."""
    args = parse_args()

    if args.camera:
        infer_live_camera(args.net, args.labels)  # Use live camera input
    else:
        images = load_input_images(args.input)
        try:
            validate_images(images, args.batch_size)
        except ValueError as e:
            logger.error(e)
            return
        infer_images(images, args.net, args.labels, args.batch_size)


if __name__ == "__main__":
    main()
