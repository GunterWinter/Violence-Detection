import argparse
import cv2
import os
import time
from pathlib import Path
from ultralytics import YOLO
import numpy as np


class PoseDetectionSystem:
    def __init__(self, opt):
        self.opt = opt
        self.model = YOLO(opt.weights)
        self.setup_directories()
        self.source_type = self.determine_source_type()
        self.vid_writer = None
        self.cap = None

    def setup_directories(self):
        self.save_dir = Path('results')
        self.save_dir.mkdir(exist_ok=True)
        (self.save_dir / 'labels').mkdir(exist_ok=True)

    def determine_source_type(self):
        if self.opt.source.isnumeric():
            return 'webcam'
        if self.opt.source.lower().startswith(('http://', 'https://')):
            return 'stream'
        if Path(self.opt.source).exists():
            if Path(self.opt.source).suffix.lower() in ['.jpg', '.png', '.jpeg']:
                return 'image'
            return 'video'
        raise ValueError("Nguồn đầu vào không hợp lệ")

    def prepare_stream_source(self):
        if 'http' in self.opt.source.lower() and not self.opt.source.lower().endswith(('.mjpeg', '.mp4')):
            return self.opt.source + '/cam.mjpeg'
        return self.opt.source

    def initialize_video_writer(self, frame):
        if self.opt.save_img:
            fps = 30 if self.source_type == 'webcam' else int(self.cap.get(cv2.CAP_PROP_FPS))
            w, h = frame.shape[1], frame.shape[0]
            output_path = str(self.save_dir / f'output_{int(time.time())}.mp4')
            self.vid_writer = cv2.VideoWriter(output_path,
                                              cv2.VideoWriter_fourcc(*'mp4v'),
                                              fps, (w, h))

    def process_image(self):
        results = self.model.predict(self.opt.source,
                                     conf=self.opt.conf,
                                     imgsz=self.opt.imgsz)
        annotated = results[0].plot()

        if self.opt.view_img:
            cv2.imshow("Pose Detection", annotated)
            cv2.waitKey(0)

        if self.opt.save_img:
            output_path = str(self.save_dir / Path(self.opt.source).name)
            cv2.imwrite(output_path, annotated)
            print(f"Ảnh đã lưu tại: {output_path}")

    def process_video_stream(self):
        if self.source_type == 'stream':
            self.opt.source = self.prepare_stream_source()

        self.cap = cv2.VideoCapture(0 if self.source_type == 'webcam' else self.opt.source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                self.handle_stream_error()
                continue

            results = self.model.track(
                frame,
                persist=True,
                conf=self.opt.conf,
                iou=self.opt.iou,
                verbose=False
            )

            annotated_frame = results[0].plot()

            if self.opt.view_img:
                cv2.imshow("Real-time Pose Detection", annotated_frame)

            if self.opt.save_img:
                if not self.vid_writer:
                    self.initialize_video_writer(annotated_frame)
                self.vid_writer.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release_resources()

    def handle_stream_error(self):
        print("Mất kết nối, đang thử kết nối lại...")
        self.cap.release()
        time.sleep(2)
        self.cap.open(self.opt.source)

    def release_resources(self):
        if self.cap:
            self.cap.release()
        if self.vid_writer:
            self.vid_writer.release()
        cv2.destroyAllWindows()

    def run(self):
        if self.source_type == 'image':
            self.process_image()
        else:
            self.process_video_stream()


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO11 Pose Detection System')
    parser.add_argument('--weights', type=str, default='yolo11n-pose.pt', help='Model path')
    parser.add_argument('--source', type=str, default='0', help='Ảnh/Video/Webcam/Stream URL')
    parser.add_argument('--imgsz', type=int, default=640, help='Kích thước inference')
    parser.add_argument('--conf', type=float, default=0.4, help='Ngưỡng confidence')
    parser.add_argument('--iou', type=float, default=0.45, help='Ngưỡng IoU cho NMS')
    parser.add_argument('--view-img', action='store_true', help='Hiển thị kết quả real-time')
    parser.add_argument('--save-img', action='store_true', help='Lưu kết quả đầu ra')
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()
    detector = PoseDetectionSystem(opt)
    detector.run()