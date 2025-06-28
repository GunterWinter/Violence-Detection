import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import argparse
import threading
import queue

class PoseDetectionSystem:
    def __init__(self, opt):
        self.opt = opt
        self.pose_model = YOLO(opt.weights)
        self.save_dir = Path('results')
        self.save_dir.mkdir(exist_ok=True)
        self.source_type = self.determine_source_type()
        self.cap = cv2.VideoCapture(self.opt.source)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video/stream source")
        self.vid_writer = None
        self.running = True
        self.frame_queue = queue.Queue(maxsize=10) if self.source_type == 'stream' else None

    def determine_source_type(self):
        if self.opt.source.lower().startswith('rtsp://'):
            return 'stream'
        elif Path(self.opt.source).exists() and Path(self.opt.source).suffix.lower() in ['.mp4', '.avi', '.mov']:
            return 'video'
        else:
            raise ValueError("Chỉ hỗ trợ RTSP streams và video files")

    def initialize_video_writer(self, frame):
        if self.opt.save and self.vid_writer is None:
            fps = 30 if self.source_type == 'stream' else int(self.cap.get(cv2.CAP_PROP_FPS))
            w, h = frame.shape[1], frame.shape[0]
            output_path = str(self.save_dir / f'output_{int(time.time())}.mp4')
            self.vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            print(f"Đã khởi tạo video writer tại: {output_path}")

    def process_video_stream(self):
        if self.opt.view:
            cv2.namedWindow("Real-time Pose Detection", cv2.WINDOW_NORMAL)
        while self.running:
            if self.source_type == 'stream':
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    if frame is None:
                        break
                else:
                    time.sleep(0.01)
                    continue
            else:
                success, frame = self.cap.read()
                if not success:
                    print("Video đã hết, thoát...")
                    break
            pose_results = self.pose_model.predict(frame, conf=self.opt.conf, imgsz=self.opt.imgsz)
            if pose_results and len(pose_results) > 0:
                frame = pose_results[0].plot()  # Sử dụng plot() để vẽ khung xương
            if self.opt.view:
                cv2.imshow("Real-time Pose Detection", frame)
            if self.opt.save:
                self.initialize_video_writer(frame)
                if self.vid_writer is not None:
                    self.vid_writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        if self.vid_writer:
            self.vid_writer.release()
            print("Đã lưu video và giải phóng VideoWriter")
        self.cap.release()
        cv2.destroyAllWindows()

    def frame_reader(self):
        """Đọc frame từ nguồn RTSP stream và đưa vào queue, xử lý lỗi OpenCV"""
        while self.running:
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.opt.source)
                if not self.cap.isOpened():
                    time.sleep(5)
                    continue
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.release()
                    time.sleep(5)
                    continue
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(frame)
            except cv2.error as e:
                print(f"OpenCV error: {e}")
                self.cap.release()
                time.sleep(5)
                continue

    def run(self):
        if self.source_type == 'stream':
            self.reader_thread = threading.Thread(target=self.frame_reader)
            self.reader_thread.start()
        self.process_video_stream()

def parse_args():
    parser = argparse.ArgumentParser(description='Pose Detection System cho RTSP và Video')
    parser.add_argument('--weights', type=str, default='yolo11n-pose.pt', help='Đường dẫn mô hình pose')
    parser.add_argument('--source', type=str, required=True, help='RTSP stream URL hoặc đường dẫn video file')
    parser.add_argument('--imgsz', type=int, default=640, help='Kích thước inference')
    parser.add_argument('--conf', type=float, default=0.4, help='Ngưỡng confidence')
    parser.add_argument('--view', action='store_true', help='Hiển thị kết quả real-time')
    parser.add_argument('--save', action='store_true', help='Lưu kết quả đầu ra liên tục')
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_args()
    detector = PoseDetectionSystem(opt)
    detector.run()