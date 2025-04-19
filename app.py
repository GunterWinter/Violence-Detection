from flask import Flask, render_template, request, Response
import threading
from pose_violence_detection import ViolencePoseDetectionSystem

app = Flask(__name__)
current_detector = None
detection_thread = None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_detection():
    global current_detector, detection_thread
    if current_detector is not None:
        return Response('Detection already running', status=400)

    rtsp_url = request.form['rtsp_url']
    enable_recording = 'enable_recording' in request.form
    record_dir = request.form['record_dir']

    # Create options object
    opt = type('Opt', (), {})()
    opt.source = rtsp_url
    opt.save = enable_recording
    opt.record_dir = record_dir
    opt.weights = 'yolo11n-pose.pt'
    opt.violence_weights = r'C:\BaiTap\Python\Violence_Detection\Yolo11_Violence_Detection\runs\detect\train\weights\best.onnx'
    opt.imgsz = 640
    opt.conf = 0.4
    opt.view = False  # No display on server
    opt.tail_length = 3

    current_detector = ViolencePoseDetectionSystem(opt)
    detection_thread = threading.Thread(target=current_detector.run)
    detection_thread.start()
    return 'Detection started'

@app.route('/stop', methods=['POST'])
def stop_detection():
    global current_detector, detection_thread
    if current_detector is None:
        return Response('No detection running', status=400)

    current_detector.running = False
    detection_thread.join()
    current_detector = None
    detection_thread = None
    return 'Detection stopped'

if __name__ == '__main__':
    app.run(debug=True)