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
import inquirer

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
    NMS_SCORE_THRESHOLD: float = 0.3
    NMS_IOU_THRESHOLD: float = 0.45
    DETECTION_COLOR: Tuple[int, int, int] = (0, 255, 0)
    FONT_SCALE: float = 0.5
    FONT_THICKNESS: int = 2
    BOX_THICKNESS: int = 2

class HeinSightSegmentationApp(GStreamerInstanceSegmentationApp):
    def __init__(self, app_callback, user_data, input_source, hef_path):
        self.input_source = input_source
        self.hef_path = hef_path
        super().__init__(app_callback, user_data)
        
    def create_pipeline(self):
        """Start with a basic pipeline using MJPG format"""
        pipeline_str = (
            f'v4l2src device={self.input_source} ! '
            'image/jpeg,width=640,height=480,framerate=30/1 ! '
            'jpegdec ! '
            'videoconvert ! '
            'video/x-raw,format=RGB ! '
            'videoconvert ! '
            'ximagesink sync=false'  # Using ximagesink instead of autovideosink
        )
        
        print(f"Creating pipeline with: {pipeline_str}")
        return Gst.parse_launch(pipeline_str)

    def _get_source_str(self):
        """Override the source string to use v4l2src"""
        return f'v4l2src device={self.input_source} ! image/jpeg,width=640,height=480,framerate=30/1 ! jpegdec ! videoconvert'

def draw_detection(
    frame: np.ndarray,
    detection: hailo.HailoDetection,
    width: int,
    height: int,
    config: DetectionConfig
) -> None:
    """Draw detection, pose, or segmentation based on task type"""
    # Add segmentation mask drawing
    mask = detection.get_mask()
    if mask is not None:
        # Resize mask to match frame size
        mask = cv2.resize(mask, (width, height))
        # Apply color overlay
        colored_mask = np.zeros_like(frame)
        colored_mask[mask > 0] = config.DETECTION_COLOR
        frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

def interactive_setup():
    """Interactive setup function to guide users through configuration"""
    
    # Find available HEF files
    hef_files = list(Path('.').glob('**/*.hef'))
    if not hef_files:
        print("❌ No HEF files found in the current directory or subdirectories!")
        return None
    
    # Create interactive questions
    questions = [
        inquirer.List('hef_file',
                     message="Which HEF file would you like to use?",
                     choices=[str(f) for f in hef_files]),
        
        inquirer.List('task_type',
                     message="What type of processing do you need?",
                     choices=[
                         ('Segmentation - for identifying object boundaries', 'segmentation'),
                         ('Pose Estimation - for tracking body positions', 'pose'),
                         ('Object Detection - for identifying objects', 'detection')
                     ]),
        
        inquirer.List('input_source',
                     message="Select your input source:",
                     choices=[
                         ('Webcam (default)', '/dev/video0'),
                         ('Other video device', 'custom_device'),
                         ('Video file', 'file')
                     ])
    ]
    
    answers = inquirer.prompt(questions)
    
    # Handle custom input source if selected
    if answers['input_source'] == 'custom_device':
        device_q = [
            inquirer.Text('device_path',
                         message="Enter the device path (e.g., /dev/video1)")
        ]
        device_answer = inquirer.prompt(device_q)
        answers['input_source'] = device_answer['device_path']
    elif answers['input_source'] == 'file':
        file_q = [
            inquirer.Text('file_path',
                         message="Enter the path to your video file")
        ]
        file_answer = inquirer.prompt(file_q)
        answers['input_source'] = file_answer['file_path']
    
    return answers

def app_callback(pad: Gst.Pad, info: Gst.Buffer, user_data: app_callback_class) -> Gst.PadProbeReturn:
    """Process inference results from the GStreamer pipeline."""
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
        detections = hailo.get_roi_from_buffer(buffer).get_objects()
        
        for detection in detections:
            if user_data.use_frame and frame is not None:
                draw_detection(frame, detection, width, height, DetectionConfig())
                
        if user_data.use_frame and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            user_data.set_frame(frame)
            
    except Exception as e:
        logger.error(f"Error processing detections: {e}")

    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    try:
        print("Welcome to HeinSight Pipeline Setup! 🚀")
        print("Let's configure your processing pipeline...\n")
        
        # Get configuration through interactive prompts
        config = interactive_setup()
        if not config:
            exit(1)
            
        print("\n✨ Starting pipeline with following configuration:")
        print(f"🔹 HEF File: {config['hef_file']}")
        print(f"🔹 Task Type: {config['task_type']}")
        print(f"🔹 Input Source: {config['input_source']}")
        
        # Create user data
        user_data = app_callback_class()
        
        # Debug logging
        print("\n🔍 Debug Information:")
        print(f"Input Source Type: {type(config['input_source'])}")
        print(f"Input Source Value: {config['input_source']}")
        
        # Create app based on task type
        if config['task_type'] == 'segmentation':
            app = HeinSightSegmentationApp(
                app_callback, 
                user_data,
                config['input_source'],
                config['hef_file']
            )
            print(f"App Input Source: {app.input_source}")
        elif config['task_type'] == 'pose':
            app = GStreamerPoseEstimationApp(app_callback, user_data)
            app.input_source = config['input_source']
            app.hef_path = config['hef_file']
        else:  # detection
            app = GStreamerDetectionApp(app_callback, user_data)
            app.input_source = config['input_source']
            app.hef_path = config['hef_file']
        
        print("\n🎥 Starting processing... Press Ctrl+C to stop.")
        app.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise