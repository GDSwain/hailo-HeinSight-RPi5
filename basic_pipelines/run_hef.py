import sys
import argparse
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp
from hailo_apps_infra.gstreamer_app import app_callback_class

def main():
    parser = argparse.ArgumentParser(description="Run a HEF model with GStreamerDetectionApp")
    parser.add_argument("--hef-path", required=True, help="Path to the HEF model file")
    parser.add_argument("--input", default="/dev/video0", help="Video input source (default: /dev/video0)")
    parser.add_argument("--use-frame", action="store_true", help="Enable frame processing")
    parser.add_argument("--show-fps", action="store_true", help="Display FPS on screen")
    parser.add_argument("--disable-sync", action="store_true", help="Disable sync for debugging")
    args = parser.parse_args()

    print(f"? Running HEF file: {args.hef_path}")

    # Create an instance of the detection app using the correct HEF file
    user_data = app_callback_class()
    from hailo_apps_infra.gstreamer_app import dummy_callback
    app = GStreamerDetectionApp(dummy_callback, user_data)
    app.hef_path = args.hef_path  # Manually override HEF path
    app.video_source = args.input  # Set the input source
    app.sync = not args.disable_sync  # Handle sync settings
    app.show_fps = args.show_fps  # Show FPS if enabled

    # Print the generated pipeline for debugging
    print("\n? Generated Pipeline:\n")
    print(app.get_pipeline_string())

    # Run the app
    app.run()

if __name__ == "__main__":
    main()
