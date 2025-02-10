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
import inquirer  # We'd need to add this dependency

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

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

class HeinSightGStreamerApp(GStreamerDetectionApp):
    # Class-level constants
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_ARCH = "hailo8l"
    
    def __init__(
        self, 
        app_callback: callable, 
        user_data: Any, 
        hef_path: str,
        config: Optional[DetectionConfig] = None
    ):
        """
        Initialize the HeinSight GStreamer Application.
        
        Args:
            app_callback: Callback function for processing frames
            user_data: User data to be passed to callback
            hef_path: Path to Hailo model file
            config: Detection configuration parameters
        """
        try:
            self.config = config or DetectionConfig()
            
            # More stringent HEF validation
            self.hef_path = Path(hef_path)
            if not self.hef_path.exists():
                raise FileNotFoundError(f"HEF file not found: {hef_path}")
            if not self.hef_path.is_file():
                raise ValueError(f"HEF path must be a file, not a directory: {hef_path}")
            if not str(self.hef_path).endswith('.hef'):
                raise ValueError(f"File must have .hef extension: {hef_path}")
            
            # Validate HEF file size
            if self.hef_path.stat().st_size < 1000:  # arbitrary minimum size
                raise ValueError(f"HEF file appears to be invalid (too small): {hef_path}")
            
            logger.info(f"Loading HEF file: {self.hef_path}")
            
            # Use the exact same path as the working version
            self.post_process_so = "/home/rogue-42/hailo-rpi5-examples/resources/libyolo_hailortpp_postprocess.so"
            
            # Log the path for debugging
            logger.info(f"Using post-process library at: {self.post_process_so}")
            
            # Check if video device exists
            if not Path("/dev/video0").exists():
                raise FileNotFoundError("Video device /dev/video0 not found")
            
            # Initialize parameters
            self.batch_size = self.DEFAULT_BATCH_SIZE
            self.arch = self.DEFAULT_ARCH
            self.post_function_name = "filter_letterbox"

            # Initialize GStreamer if not already initialized
            if not Gst.is_initialized():
                Gst.init(None)

            super().__init__(app_callback, user_data)
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def get_pipeline_string(self) -> str:
        """
        Creates GStreamer pipeline with configured elements.
        
        Returns:
            str: GStreamer pipeline configuration string
        """
        logger.info("Creating GStreamer pipeline...")
        
        try:
            # Determine input source element
            if self.input_source.startswith('/dev/'):
                source_element = f"v4l2src device={self.input_source}"
            else:
                source_element = f"filesrc location={self.input_source} ! decodebin"
            
            pipeline = (
                f"{source_element} ! "
                f"video/x-raw, width=1280, height=720 ! "
                f"videoconvert ! "
                f"videoscale ! "
                f"video/x-raw, width=640, height=640, format=RGB ! "
                f"hailonet hef-path={self.hef_path} batch-size={self.batch_size} debug=true ! "
                f"hailofilter so-path={self.post_process_so} function-name={self.post_function_name} ! "
                f"hailooverlay ! "
                f"identity name=identity_callback ! "
                f"videoconvert ! "
                f"fpsdisplaysink name=hailo_display video-sink=autovideosink sync=false text-overlay=true signal-fps-measurements=true"
            )
            
            logger.info(f"Pipeline string: {pipeline}")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error creating pipeline string: {e}")
            raise

def draw_detection(
    frame: np.ndarray,
    detection: hailo.HailoDetection,
    width: int,
    height: int,
    config: DetectionConfig
) -> None:
    """
    Draw detection boxes and labels on the frame.
    
    Args:
        frame: Input frame to draw on
        detection: Hailo detection object
        width: Frame width
        height: Frame height
        config: Detection configuration parameters
    """
    bbox = detection.get_bbox()
    x1, y1 = int(bbox.xmin() * width), int(bbox.ymin() * height)
    x2, y2 = int(bbox.xmax() * width), int(bbox.ymax() * height)
    
    # Draw bounding box
    cv2.rectangle(
        frame, 
        (x1, y1), 
        (x2, y2), 
        config.DETECTION_COLOR, 
        config.BOX_THICKNESS
    )
    
    # Draw label
    label_text = f"{detection.get_label()} {detection.get_confidence():.2f}"
    cv2.putText(
        frame,
        label_text,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.FONT_SCALE,
        config.DETECTION_COLOR,
        config.FONT_THICKNESS
    )

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
        detections = hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION)
        
        for detection in detections:
            label = detection.get_label()
            confidence = detection.get_confidence()
            
            if label == "vial":
                logger.info(f"Vial detected: Confidence {confidence:.2f}")

            if user_data.use_frame and frame is not None:
                draw_detection(frame, detection, width, height, DetectionConfig())
                
        if user_data.use_frame and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            user_data.set_frame(frame)
            
    except Exception as e:
        logger.error(f"Error processing detections: {e}")

    return Gst.PadProbeReturn.OK

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
        
        # Create user data and app as before
        user_data = app_callback_class()
        app = HeinSightGStreamerApp.create_app(
            hef_path=config['hef_file'],
            app_callback=app_callback,
            user_data=user_data
        )
        
        print("\n🎥 Starting processing... Press Ctrl+C to stop.")
        app.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise