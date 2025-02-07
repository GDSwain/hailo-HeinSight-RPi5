import subprocess
import re
import sys

def get_hef_properties(hef_path):
    """Extracts model properties from a .hef file."""
    cmd = ["hailortcli", "parse-hef", "--parse-vstreams", hef_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error parsing HEF file:", result.stderr)
        return None

    input_pattern = re.compile(r"Input\s+(\S+)\s+(\S+),\s+NHWC\((\d+)x(\d+)x(\d+)\)")
    output_pattern = re.compile(r"Output\s+(\S+)\s+(\S+),\s+FCR\((\d+)x(\d+)x(\d+)\)")

    inputs, outputs = [], []

    for line in result.stdout.split("\n"):
        if (match := input_pattern.search(line)):
            name, dtype, h, w, c = match.groups()
            inputs.append({"name": name, "dtype": dtype, "shape": (int(h), int(w), int(c))})

        if (match := output_pattern.search(line)):
            name, dtype, h, w, c = match.groups()
            outputs.append({"name": name, "dtype": dtype, "shape": (int(h), int(w), int(c))})

    return {"inputs": inputs, "outputs": outputs}

def build_gst_pipeline(hef_path, camera_device="/dev/video0"):
    """Builds a dynamic GStreamer pipeline based on the HEF model."""
    hef_properties = get_hef_properties(hef_path)
    
    if not hef_properties:
        return None
    
    input_shape = hef_properties["inputs"][0]["shape"]
    width, height, _ = input_shape  # Extract width/height
    
    # Correct path to libyolo_hailortpp_postprocess.so
    hailo_filter_path = "/home/rogue-42/hailo-rpi5-examples/venv_hailo_rpi5_examples/lib/python3.11/site-packages/resources/libyolo_hailortpp_postprocess.so"

    # Construct GStreamer pipeline
    gst_pipeline = f"""
    v4l2src device={camera_device} ! video/x-raw, width=1280, height=720 ! 
    videoscale ! video/x-raw, width={width}, height={height} ! videoconvert ! 
    hailonet hef-path={hef_path} batch-size=1 ! 
    hailofilter so-path={hailo_filter_path} function-name=filter_letterbox ! 
    hailooverlay ! 
    autovideosink
    """

    return gst_pipeline.strip()

    # Add post-processing if the model has detection outputs
    has_nms = any("conv8" in out["name"] for out in hef_properties["outputs"])
    
    if has_nms:
        gst_pipeline += """
        hailofilter so-path=/home/rogue-42/hailo-rpi5-examples/resources/libyolo_hailortpp_postprocess.so function-name=filter_letterbox ! 
        hailooverlay ! 
        """
    
    # Add final output
    gst_pipeline += "autovideosink"

    return gst_pipeline.strip()

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_hef.py <path_to_hef> [camera_device]")
        sys.exit(1)

    hef_path = sys.argv[1]
    camera_device = sys.argv[2] if len(sys.argv) > 2 else "/dev/video0"

    print(f"Parsing HEF file: {hef_path}")
    gst_pipeline = build_gst_pipeline(hef_path, camera_device)

    if not gst_pipeline:
        print("Error: Could not generate GStreamer pipeline.")
        sys.exit(1)

    print("\nGenerated GStreamer Pipeline:\n", gst_pipeline)
    print("\nLaunching pipeline...\n")

    subprocess.run(["gst-launch-1.0"] + gst_pipeline.split())

if __name__ == "__main__":
    main()
