import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

class HeinSightGStreamerApp(GStreamerDetectionApp):
    def __init__(self, app_callback, user_data, hef_path):
        self.hef_path = hef_path  # Hailo Model Path
        self.batch_size = 1  # Set batch size for inference
        self.arch = "hailo8l"  # Since we now know the correct architecture

        # Define NMS settings (modify as needed)
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45

        # Set the Hailo inference and post-processing
        self.post_process_so = "/home/rogue-42/hailo-rpi5-examples/resources/libyolo_hailortpp_postprocess.so"
        self.post_function_name = "filter_letterbox"

        # Call parent class (GStreamerDetectionApp)
        super().__init__(app_callback, user_data)

def get_pipeline_string(self):
    """Creates GStreamer pipeline with missing elements fixed"""
    pipeline = f"""
    v4l2src device=/dev/video0 ! video/x-raw, width=1280, height=720 ! \
    videoconvert ! videoscale ! video/x-raw, width=640, height=640 ! \
    hailonet hef-path={self.hef_path} batch-size={self.batch_size} ! \
    hailofilter so-path={self.post_process_so} function-name={self.post_function_name} ! \
    hailooverlay ! identity name=identity_callback ! \
    fpsdisplaysink name=hailo_display video-sink=autovideosink sync=false text-overlay=true signal-fps-measurements=true
    """
    return pipeline


def app_callback(pad, info, user_data):
    """
    Process inference results.
    """
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Count Frames
    user_data.increment()
    frame_count = user_data.get_count()
    print(f"Frame count: {frame_count}")

    # Get metadata from inference
    format, width, height = get_caps_from_pad(pad)
    frame = None

    if user_data.use_frame and format:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    detections = hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION)
    
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()

        if label == "vial":
            print(f"Vial detected: Confidence {confidence:.2f}")

        if user_data.use_frame:
            # Draw detection
            x1, y1 = int(bbox.xmin() * width), int(bbox.ymin() * height)
            x2, y2 = int(bbox.xmax() * width), int(bbox.ymax() * height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    user_data = app_callback_class()
    hef_path = "/home/rogue-42/hailo-rpi5-examples/resources/vessel_h8l.hef"  # UPDATE WHEN READY
    app = HeinSightGStreamerApp(app_callback, user_data, hef_path)
    app.run()
