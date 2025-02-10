import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Any
from pathlib import Path
import argparse

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp
from hailo_apps_infra.instance_segmentation_pipeline import GStreamerInstanceSegmentationApp
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DetectionConfig:
    """Configuration for detection parameters"""
    TASK_TYPE: str = "detection"  # Options: "detection", "pose", "segmentation"
    NMS_SCORE_THRESHOLD: float = 0.3
    NMS_IOU_THRESHOLD: float = 0.45
    DETECTION_COLOR: Tuple[int, int, int] = (0, 255, 0)
    FONT_SCALE: float = 0.5
    FONT_THICKNESS: int = 2
    BOX_THICKNESS: int = 2

class HeinSightGStreamerApp:
    @staticmethod
    def determine_task_type(hef_path: str) -> str:
        """Determine task type based on HEF file name"""
        hef_name = Path(hef_path).name.lower()
        if 'pose' in hef_name:
            return 'pose'
        elif 'seg' in hef_name:
            return 'segmentation'
        else:
            return 'detection'

    @staticmethod
    def validate_hef_architecture(hef_path: str) -> bool:
        """Validate that the HEF file matches the device architecture"""
        hef_name = Path(hef_path).name.lower()
        # Check if the HEF file is specifically for HAILO8L
        return 'h8l' in hef_name

    @staticmethod
    def create_app(hef_path: str, app_callback: callable, user_data: Any, **kwargs) -> Any:
        """Factory method to create the appropriate app based on HEF file"""
        # Validate HEF architecture
        if not HeinSightGStreamerApp.validate_hef_architecture(hef_path):
            logger.warning(f"HEF file {hef_path} might not be compatible with HAILO8L architecture")
            
        task_type = HeinSightGStreamerApp.determine_task_type(hef_path)
        logger.info(f"Creating app for task type: {task_type}")
        
        if task_type == "detection":
            return GStreamerDetectionApp(app_callback, user_data)
        elif task_type == "segmentation":
            return GStreamerInstanceSegmentationApp(app_callback, user_data)
        elif task_type == "pose":
            return GStreamerPoseEstimationApp(app_callback, user_data)
        else:
            raise ValueError(f"Unknown task type for HEF: {hef_path}")

def draw_detection(
    frame: np.ndarray,
    detection: hailo.HailoDetection,
    width: int,
    height: int,
    config: DetectionConfig
) -> None:
    """Draw detection, pose, or segmentation based on task type"""
    if config.TASK_TYPE == "detection":
        # Existing detection drawing code
        bbox = detection.get_bbox()
        x1, y1 = int(bbox.xmin() * width), int(bbox.ymin() * height)
        x2, y2 = int(bbox.xmax() * width), int(bbox.ymax() * height)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), config.DETECTION_COLOR, config.BOX_THICKNESS)
        label_text = f"{detection.get_label()} {detection.get_confidence():.2f}"
        cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                   config.FONT_SCALE, config.DETECTION_COLOR, config.FONT_THICKNESS)
    
    elif config.TASK_TYPE == "pose":
        # Add pose estimation drawing
        landmarks = detection.get_landmarks()
        for landmark in landmarks:
            x, y = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(frame, (x, y), 3, config.DETECTION_COLOR, -1)
        # Draw connections between landmarks if needed
        
    elif config.TASK_TYPE == "segmentation":
        # Add segmentation mask drawing
        mask = detection.get_mask()
        if mask is not None:
            # Resize mask to match frame size
            mask = cv2.resize(mask, (width, height))
            # Apply color overlay
            colored_mask = np.zeros_like(frame)
            colored_mask[mask > 0] = config.DETECTION_COLOR
            frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

def app_callback(pad: Gst.Pad, info: Gst.Buffer, user_data: app_callback_class) -> Gst.PadProbeReturn:
    """
    Process inference results from the GStreamer pipeline.
    
    Args:
        pad: GStreamer pad
        info: Buffer containing frame data
        user_data: User data object containing frame information
        
    Returns:
        GStreamer pad probe return value
    """
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Update frame counter
    user_data.increment()
    logger.debug(f"Processing frame: {user_data.get_count()}")

    # Get frame metadata
    format, width, height = get_caps_from_pad(pad)
    frame = None

    if user_data.use_frame and format:
        try:
            frame = get_numpy_from_buffer(buffer, format, width, height)
        except Exception as e:
            logger.error(f"Error getting frame from buffer: {e}")
            return Gst.PadProbeReturn.OK

    # Process detections
    try:
        if config.TASK_TYPE == "detection":
            detections = hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION)
        elif config.TASK_TYPE == "pose":
            detections = hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_LANDMARKS)
        elif config.TASK_TYPE == "segmentation":
            detections = hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_SEGMENTATION)
        
        for detection in detections:
            label = detection.get_label()
            confidence = detection.get_confidence()
            
            if label == "vial":
                logger.info(f"Vial detected: Confidence {confidence:.2f}")

            if user_data.use_frame and frame is not None:
                draw_detection(frame, detection, width, height, config)
                
        if user_data.use_frame and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            user_data.set_frame(frame)
            
    except Exception as e:
        logger.error(f"Error processing detections: {e}")

    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    try:
        # Use the base argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', type=str, default='/dev/video0',
                          help='Input source (video device or file)')
        parser.add_argument('--use-frame', action='store_true',
                          help='Use frame processing')
        parser.add_argument('--show-fps', action='store_true',
                          help='Show FPS counter')
        parser.add_argument('--arch', choices=['hailo8', 'hailo8l'],
                          help='Hailo architecture')
        parser.add_argument('--hef-path', type=str, required=True,
                          help='Path to HEF model file')
        parser.add_argument('--disable-sync', action='store_true',
                          help='Disable synchronization')
        parser.add_argument('--disable-callback', action='store_true',
                          help='Disable callback')
        parser.add_argument('--dump-dot', action='store_true',
                          help='Dump DOT file')
        parser.add_argument('--labels-json', type=str,
                          help='Path to labels JSON file')
        
        args = parser.parse_args()
        
        user_data = app_callback_class()
        
        # Create app based on HEF file
        app = HeinSightGStreamerApp.create_app(
            hef_path=args.hef_path,
            app_callback=app_callback,
            user_data=user_data
        )
        
        app.run()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise