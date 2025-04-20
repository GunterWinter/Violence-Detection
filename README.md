# Violence Detection System

## Overview

This project is a **Violence Detection System** that utilizes computer vision and machine learning to detect violent activities and poses in real-time from various sources such as RTSP streams, uploaded images, or videos. It leverages YOLO models for violence detection and pose estimation, and provides a web-based interface for user interaction built with Flask and Bootstrap.

## Features

- **Real-time Violence Detection**: Identifies violent activities from RTSP streams or webcam feeds.
- **Pose Estimation**: Detects falling or abnormal poses to enhance the identification of serious violence.
- **Web Interface**: A user-friendly interface to start/stop detection and upload files.
- **Recording**: Automatically records video streams when violence is detected.
- **File Upload**: Supports uploading images or videos for offline violence detection.

## Requirements

- **Python 3.8+**
- **Flask**
- **OpenCV**
- **Ultralytics YOLO**
- **FFmpeg** (for recording RTSP streams)
- **Torch** (with CUDA support recommended for GPU acceleration)

## Installation

1. **Clone the Repository**:

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLO Models**:

   - Download the pose detection model (`yolo11n-pose.pt`) and violence detection model (`best.onnx`) and place them in the directories specified in the code (e.g., `weights` and `violence_weights` paths).

4. **FFmpeg Setup**:

   - Install FFmpeg on your system to enable recording of RTSP streams. Ensure it is accessible from the command line.

## Usage

1. **Run the Application**:

   ```bash
   python app.py
   ```

2. **Access the Web Interface**:

   - Open a web browser and go to `http://localhost:5000`.

3. **RTSP Streaming**:

   - In the "RTSP Streaming" tab, enter the RTSP details (admin, password, IP, port, path, protocol).
   - Click "Start Detection" to begin real-time violence detection.
   - Check "Enable recording" to save video clips when violence is detected.

4. **File Upload**:

   - In the "Upload File" tab, select an image or video file.
   - Click "Start Detection" to process the file offline.
   - View or download the processed results from the interface.

5. **Stop Detection**:

   - For RTSP streams, click "Stop Detection" to halt the process.

## Configuration Options

- **Image Size (**`imgsz`**)**: Adjust the inference image size (default: 640 pixels).
- **Confidence Threshold (**`conf`**)**: Set the detection confidence level (default: 0.4).
- **Tail Length**: Duration (in seconds) to continue recording after violence is no longer detected (default: 5).
- **Recording Directory**: Specify where to save recorded videos or processed files (default: `recordings`).

## Project Structure

- `app.py`: Main Flask application handling routes and detection logic.
- `pose_violence_detection.py`: Core detection logic using YOLO models.
- `templates/index.html`: Web interface template for user interaction.
- `uploads/`: Directory for uploaded files.
- `output/`: Directory for processed results (used if not overridden).

## Notes

- Ensure the RTSP stream URL is correctly formatted and accessible (e.g., `rtsp://admin:password@ip:port/path`).
- For optimal performance, use a GPU with CUDA support for model inference.
- Performance may vary depending on hardware capabilities, especially for real-time streams.

## License

This project is licensed under the MIT License.
