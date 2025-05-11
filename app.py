from flask import Flask, render_template, request, Response, send_from_directory
import threading
from pose_violence_detection import ViolencePoseDetectionSystem
import os
import cv2
import time

app = Flask(__name__)
current_detector = None
detection_thread = None

# Tạo thư mục uploads và output nếu chưa tồn tại
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/', methods=['GET'])
def index():
    """Render trang chủ với giao diện người dùng."""
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_detection():
    """Bắt đầu quá trình phát hiện bạo lực từ RTSP stream."""
    global current_detector, detection_thread
    if current_detector is not None:
        return Response('Detection already running', status=400)

    # Lấy các giá trị từ form để xây dựng URL RTSP
    admin = request.form['admin']
    password = request.form['password']
    ip = request.form['ip']
    port = request.form['port']
    path = request.form['path']
    protocol = request.form['protocol']  # 'tcp' hoặc 'udp'

    # Xây dựng URL RTSP đúng cú pháp
    rtsp_url = f"rtsp://{admin}:{password}@{ip}:{port}/{path}?rtsp_transport={protocol}"

    record_dir = request.form['record_dir']

    # Tạo dictionary cho options thay vì dùng type()
    opt = {
        'source': rtsp_url,
        'save': False,  # Ghi hình sẽ được điều khiển thủ công
        'record_dir': record_dir,
        'weights': 'yolo11n-pose.pt',
        'violence_weights': r'C:\BaiTap\Python\Violence_Detection\Yolo11_Violence_Detection\runs\detect\train\weights\best.onnx',
        'imgsz': int(request.form['imgsz']),
        'conf': float(request.form['conf']),
        'view': False,  # Không hiển thị trên server
        'tail_length': int(request.form['tail_length'])
    }

    current_detector = ViolencePoseDetectionSystem(opt)
    detection_thread = threading.Thread(target=current_detector.run)
    detection_thread.start()
    return 'Detection started'

@app.route('/stop', methods=['POST'])
def stop_detection():
    """Dừng quá trình phát hiện bạo lực."""
    global current_detector, detection_thread
    if current_detector is None:
        return Response('No detection running', status=400)

    if current_detector.recording:
        current_detector.stop_recording()
    current_detector.running = False
    detection_thread.join()
    current_detector = None
    detection_thread = None
    return 'Detection stopped'

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """Bắt đầu ghi hình."""
    global current_detector
    if current_detector is None or not current_detector.running:
        return Response('Detection not running', status=400)
    if current_detector.recording:
        return Response('Already recording', status=400)
    current_detector.start_recording()
    return 'Recording started'

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """Dừng ghi hình."""
    global current_detector
    if current_detector is None or not current_detector.running:
        return Response('Detection not running', status=400)
    if not current_detector.recording:
        return Response('Not recording', status=400)
    current_detector.stop_recording()
    return 'Recording stopped'

@app.route('/upload', methods=['POST'])
def upload_file():
    """Xử lý upload file (ảnh hoặc video) và chạy phát hiện bạo lực."""
    file = request.files['file']
    if file:
        filename = file.filename
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Tạo dictionary cho options
        opt = {
            'source': file_path,
            'save': True,
            'record_dir': request.form['record_dir'],
            'weights': 'yolo11n-pose.pt',
            'violence_weights': r'C:\BaiTap\Python\Violence_Detection\Yolo11_Violence_Detection\runs\detect\train\weights\best.pt',
            'imgsz': int(request.form['imgsz']),
            'conf': float(request.form['conf']),
            'view': False,
            'tail_length': 3  # just for fun
        }

        detector = ViolencePoseDetectionSystem(opt)
        detector.run()
        return {'type': 'image' if filename.endswith(('.jpg', '.png', '.jpeg')) else 'video', 'filename': detector.output_path}
    return {'error': 'No file uploaded'}, 400

@app.route('/output/<path:filename>')
def serve_output(filename):
    """Phục vụ file đầu ra từ thư mục output."""
    if current_detector and current_detector.output_dir:
        return send_from_directory(current_detector.output_dir, filename)
    else:
        return send_from_directory('output', filename)

def gen_frames():
    """Tạo frame cho video stream từ detector."""
    while True:
        if current_detector is not None:
            frame = current_detector.get_frame()  # Lấy frame từ detector
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    """Route để stream video feed."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)