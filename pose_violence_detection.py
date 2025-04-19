import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import torch
import numpy as np
from datetime import datetime
import threading
import queue
from ffmpeg import FFmpeg

class ViolencePoseDetectionSystem:
    def __init__(self, opt):
        self.opt = opt
        self.violence_model = YOLO(opt.violence_weights)
        self.pose_model = YOLO(opt.weights)
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
        self.source_type = self.determine_source_type()
        self.cap = None
        self.vid_writer = None
        self.save = opt.save
        self.tail_length = opt.tail_length
        self.recording = False
        self.ffmpeg_process = None
        self.activity_count = 0
        self.tail_frames = None
        self.running = True
        # Chỉ dùng thread cho stream và webcam
        self.use_thread = self.source_type in ['stream', 'webcam']
        self.frame_queue = queue.Queue(maxsize=10) if self.use_thread else None

        if self.source_type != 'image':
            self.cap = cv2.VideoCapture(0 if self.source_type == 'webcam' else self.opt.source)
            if not self.cap.isOpened():
                raise ValueError("Không thể mở nguồn video/stream")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
            self.tail_frames = self.tail_length * self.fps

    def determine_source_type(self):
        if self.opt.source.isnumeric():
            return 'webcam'
        if self.opt.source.lower().startswith(('http://', 'https://', 'rtsp://')):
            return 'stream'
        if Path(self.opt.source).exists():
            if Path(self.opt.source).suffix.lower() in ['.jpg', '.png', '.jpeg']:
                return 'image'
            return 'video'
        raise ValueError("Nguồn đầu vào không hợp lệ")

    def frame_reader(self):
        while self.running:
            if self.source_type == 'stream' and self.opt.source.startswith('rtsp://'):
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.opt.source)
                    if not self.cap.isOpened():
                        time.sleep(5)
                        continue
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(frame)
            else:
                if self.source_type == 'video':
                    self.frame_queue.put(None)
                    break
                elif self.source_type == 'stream':
                    self.cap.release()
                    time.sleep(5)
                else:
                    time.sleep(0.1)

    def crop_image(self, image, box):
        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        return image[ymin:ymax, xmin:xmax], (xmin, ymin)

    def draw_skeleton(self, image, kpts, offset):
        if kpts is None or len(kpts.data) == 0:
            return
        for person_kpts in kpts.data:
            for kpt in person_kpts:
                if kpt[2] > 0:
                    x, y = int(kpt[0] + offset[0]), int(kpt[1] + offset[1])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    def is_falling(self, person_kpts, person_box, frame_height):
        left_shoulder_y = person_kpts[5][1] if person_kpts[5][2] > 0 else None
        right_shoulder_y = person_kpts[6][1] if person_kpts[6][2] > 0 else None
        shoulder_y = None
        if left_shoulder_y and right_shoulder_y:
            shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        elif left_shoulder_y:
            shoulder_y = left_shoulder_y
        elif right_shoulder_y:
            shoulder_y = right_shoulder_y
        if not shoulder_y:
            return False
        xmin, ymin, xmax, ymax = person_box
        dx, dy = xmax - xmin, ymax - ymin
        difference = dy - dx
        thre = (frame_height // 2) + 100
        return (difference <= 0 and shoulder_y > thre) or (difference < 0)

    def initialize_video_writer(self, frame):
        if self.vid_writer is None:
            fps = 30 if self.source_type == 'webcam' else self.cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30  # Đặt mặc định nếu không lấy được FPS
            w, h = frame.shape[1], frame.shape[0]
            output_path = str(self.output_dir / f'output_{int(time.time())}.mp4')
            self.vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            print(f"Đã khởi tạo video writer tại: {output_path}")

    def start_recording(self):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = self.output_dir / f'recording_{timestamp}.mkv'
        self.ffmpeg_process = (
            FFmpeg()
            .option("y")
            .input(self.opt.source, rtsp_transport="tcp", rtsp_flags="prefer_tcp")
            .output(str(filename), vcodec="copy", acodec="copy")
        )
        self.ffmpeg_thread = threading.Thread(target=self.ffmpeg_process.execute)
        self.ffmpeg_thread.start()
        self.recording = True
        print(f"Bắt đầu ghi hình: {filename}")

    def stop_recording(self):
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.ffmpeg_thread.join()
            self.ffmpeg_process = None
            self.recording = False
            print("Đã dừng ghi hình")

    def process_image(self):
        image = cv2.imread(self.opt.source)
        violence_results = self.violence_model.predict(image, conf=self.opt.conf, imgsz=self.opt.imgsz, verbose=False)
        violence_class_id = next(k for k, v in self.violence_model.names.items() if v == 'Violence')
        violence_boxes = [box for box in violence_results[0].boxes if box.cls == violence_class_id]

        for box in violence_boxes:
            cropped_image, offset = self.crop_image(image, box)
            pose_results = self.pose_model.predict(cropped_image, conf=self.opt.conf, imgsz=self.opt.imgsz, verbose=False)
            p1 = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
            p2 = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
            cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
            cv2.putText(image, "Violence", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if pose_results and pose_results[0].keypoints is not None:
                self.draw_skeleton(image, pose_results[0].keypoints, offset)

        if self.opt.view:
            cv2.imshow("Violence Detection", image)
            cv2.waitKey(0)
        if self.save:
            output_path = str(self.output_dir / Path(self.opt.source).name)
            cv2.imwrite(output_path, image)
            print(f"Ảnh đã lưu tại: {output_path}")

    def process_video_directly(self):
        """Xử lý video tuần tự không dùng thread"""
        self.cap = cv2.VideoCapture(self.opt.source)
        if not self.cap.isOpened():
            raise ValueError("Không thể mở video")
        if self.opt.view:
            cv2.namedWindow("Violence Detection", cv2.WINDOW_NORMAL)

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            frame_height = frame.shape[0]
            violence_results = self.violence_model.predict(frame, conf=self.opt.conf, imgsz=self.opt.imgsz, verbose=False)
            violence_class_id = next(k for k, v in self.violence_model.names.items() if v == 'Violence')
            violence_boxes = [box for box in violence_results[0].boxes if box.cls == violence_class_id]

            for box in violence_boxes:
                cropped_frame, offset = self.crop_image(frame, box)
                pose_results = self.pose_model.predict(cropped_frame, conf=self.opt.conf, imgsz=self.opt.imgsz, verbose=False)
                p1 = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
                p2 = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
                label = "Violence"
                color = (0, 255, 0)
                if pose_results and pose_results[0].keypoints is not None and pose_results[0].boxes is not None:
                    kpts = pose_results[0].keypoints.data
                    boxes = pose_results[0].boxes.xyxy
                    if boxes.shape[0] > 0:
                        for i in range(kpts.shape[0]):
                            if i < boxes.shape[0]:
                                person_kpts = kpts[i]
                                person_box = boxes[i].clone()
                                person_box[0] += offset[0]
                                person_box[1] += offset[1]
                                person_box[2] += offset[0]
                                person_box[3] += offset[1]
                                if self.is_falling(person_kpts, person_box, frame_height):
                                    xmin, ymin, xmax, ymax = person_box.int().tolist()
                                    cv2.putText(frame, "Falling", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                                (0, 0, 255), 2)
                                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                                    label = "Serious Violence"
                                    color = (0, 0, 255)
                cv2.rectangle(frame, p1, p2, color, 2)
                cv2.putText(frame, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                if pose_results and pose_results[0].keypoints is not None:
                    self.draw_skeleton(frame, pose_results[0].keypoints, offset)

            if self.save:
                if self.vid_writer is None:
                    self.initialize_video_writer(frame)
                self.vid_writer.write(frame)

            if self.opt.view:
                cv2.imshow("Violence Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if self.vid_writer:
            self.vid_writer.release()
        self.cap.release()
        cv2.destroyAllWindows()

    def process_video_stream(self):
        """Xử lý stream và webcam bằng thread"""
        if self.opt.view:
            cv2.namedWindow("Real-time Violence Detection", cv2.WINDOW_NORMAL)

        self.reader_thread = threading.Thread(target=self.frame_reader)
        self.reader_thread.start()

        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                if frame is None:
                    break
                frame_height = frame.shape[0]
                violence_results = self.violence_model.predict(frame, conf=self.opt.conf, imgsz=self.opt.imgsz, verbose=False)
                violence_class_id = next(k for k, v in self.violence_model.names.items() if v == 'Violence')
                violence_boxes = [box for box in violence_results[0].boxes if box.cls == violence_class_id]
                violence_detected = len(violence_boxes) > 0

                for box in violence_boxes:
                    cropped_frame, offset = self.crop_image(frame, box)
                    pose_results = self.pose_model.predict(cropped_frame, conf=self.opt.conf, imgsz=self.opt.imgsz, verbose=False)
                    p1 = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
                    p2 = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
                    label = "Violence"
                    color = (0, 255, 0)
                    if pose_results and pose_results[0].keypoints is not None and pose_results[0].boxes is not None:
                        kpts = pose_results[0].keypoints.data
                        boxes = pose_results[0].boxes.xyxy
                        if boxes.shape[0] > 0:
                            for i in range(kpts.shape[0]):
                                if i < boxes.shape[0]:
                                    person_kpts = kpts[i]
                                    person_box = boxes[i].clone()
                                    person_box[0] += offset[0]
                                    person_box[1] += offset[1]
                                    person_box[2] += offset[0]
                                    person_box[3] += offset[1]
                                    if self.is_falling(person_kpts, person_box, frame_height):
                                        xmin, ymin, xmax, ymax = person_box.int().tolist()
                                        cv2.putText(frame, "Falling", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                                    (0, 0, 255), 2)
                                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                                        label = "Serious Violence"
                                        color = (0, 0, 255)
                    cv2.rectangle(frame, p1, p2, color, 2)
                    cv2.putText(frame, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    if pose_results and pose_results[0].keypoints is not None:
                        self.draw_skeleton(frame, pose_results[0].keypoints, offset)

                if self.save:
                    if self.source_type == 'stream' and self.opt.source.startswith('rtsp://'):
                        if violence_detected:
                            if not self.recording:
                                self.start_recording()
                            self.activity_count = 0
                        elif self.recording:
                            self.activity_count += 1
                            if self.activity_count > self.tail_frames:
                                self.stop_recording()
                    elif self.source_type == 'webcam':
                        if self.vid_writer is None:
                            self.initialize_video_writer(frame)
                        self.vid_writer.write(frame)

                if self.opt.view:
                    cv2.imshow("Real-time Violence Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
            else:
                time.sleep(0.01)

        if self.vid_writer:
            self.vid_writer.release()
        if self.recording:
            self.stop_recording()
        self.running = False
        self.reader_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        if self.source_type == 'image':
            self.process_image()
        elif self.use_thread:
            self.process_video_stream()
        else:
            self.process_video_directly()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Violence and Pose Detection System')
    parser.add_argument('--weights', type=str, default='yolo11n-pose.pt', help='Đường dẫn mô hình pose')
    parser.add_argument('--violence-weights', type=str,
                        default=r'C:\BaiTap\Python\Violence_Detection\Yolo11_Violence_Detection\runs\detect\train\weights\best.onnx',
                        help='Đường dẫn mô hình Violence')
    parser.add_argument('--source', type=str, default='0', help='Ảnh/Video/Webcam/Stream URL')
    parser.add_argument('--imgsz', type=int, default=640, help='Kích thước inference')
    parser.add_argument('--conf', type=float, default=0.4, help='Ngưỡng confidence')
    parser.add_argument('--view', action='store_true', help='Hiển thị kết quả real-time')
    parser.add_argument('--save', action='store_true', help='Lưu kết quả đầu ra hoặc ghi hình nếu là RTSP stream')
    parser.add_argument('--tail_length', type=int, default=5,
                        help='Thời gian (giây) tiếp tục ghi hình sau khi không còn phát hiện bạo lực')
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_args()
    detector = ViolencePoseDetectionSystem(opt)
    detector.run()