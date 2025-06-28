import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import argparse
import threading
import queue
import torch

class PoseDetectionSystem:
    def __init__(self, opt):
        self.opt = opt
        self.pose_model = YOLO(opt.weights)
        self.save_dir = Path('results')
        self.save_dir.mkdir(exist_ok=True)
        self.source_type = self.determine_source_type()
        self.cap = None
        self.vid_writer = None
        self.running = True
        self.frame_queue = queue.Queue(maxsize=10) if self.source_type in ['stream', 'webcam'] else None

        if self.source_type != 'image':
            self.cap = cv2.VideoCapture(0 if self.source_type == 'webcam' else self.opt.source)
            if not self.cap.isOpened():
                raise ValueError("Unable to open video/stream source")

    def determine_source_type(self):
        if self.opt.source.isnumeric():
            return 'webcam'
        if self.opt.source.lower().startswith('rtsp://'):
            return 'stream'
        if Path(self.opt.source).exists():
            if Path(self.opt.source).suffix.lower() in ['.jpg', '.png', '.jpeg']:
                return 'image'
            return 'video'
        raise ValueError("Nguồn đầu vào không hợp lệ")

    def draw_skeleton(self, image, kpts):
        """Vẽ khung xương lên ảnh gốc với các đường nối giữa các keypoint"""
        if kpts is None or len(kpts.data) == 0:
            return
        pairs = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Đầu và vai
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Tay
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Chân
            (5, 11), (6, 12)  # Kết nối thân
        ]
        for person_kpts in kpts.data:
            if len(person_kpts) == 0:
                continue
            person_kpts_np = person_kpts.cpu().numpy()
            for pair in pairs:
                if pair[0] < len(person_kpts_np) and pair[1] < len(person_kpts_np):
                    pt1 = person_kpts_np[pair[0]]
                    pt2 = person_kpts_np[pair[1]]
                    if pt1[2] > 0.5 and pt2[2] > 0.5:
                        x1, y1 = int(pt1[0]), int(pt1[1])
                        x2, y2 = int(pt2[0]), int(pt2[1])
                        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for kpt in person_kpts_np:
                if kpt[2] > 0.5:
                    x, y = int(kpt[0]), int(kpt[1])
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    def is_falling(self, person_kpts, person_box, frame_height):
        """Kiểm tra xem một người có đang té ngã hay không"""
        person_kpts_np = person_kpts.cpu().numpy()
        required_indices = [5, 6, 11, 12]
        if all(person_kpts_np[i, 2] > 0.5 for i in required_indices):
            shoulder_mid = np.array([(person_kpts_np[5, 0] + person_kpts_np[6, 0]) / 2,
                                    (person_kpts_np[5, 1] + person_kpts_np[6, 1]) / 2])
            hip_mid = np.array([(person_kpts_np[11, 0] + person_kpts_np[12, 0]) / 2,
                               (person_kpts_np[11, 1] + person_kpts_np[12, 1]) / 2])
            vector = hip_mid - shoulder_mid
            angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi
            angle_deviation = abs(angle - 90)
            if angle_deviation > 45:
                return True
            xmin, ymin, xmax, ymax = person_box
            if (xmax - xmin) > (ymax - ymin):
                return True
        return False

    def is_crawling(self, person_kpts, person_box, frame_height):
        """Kiểm tra xem một người có đang bò hay không"""
        person_kpts_np = person_kpts.cpu().numpy()
        required_indices = [5, 6, 11, 12, 9, 10, 15, 16]
        if all(person_kpts_np[i, 2] > 0.5 for i in required_indices):
            shoulder_mid = np.array([(person_kpts_np[5, 0] + person_kpts_np[6, 0]) / 2,
                                    (person_kpts_np[5, 1] + person_kpts_np[6, 1]) / 2])
            hip_mid = np.array([(person_kpts_np[11, 0] + person_kpts_np[12, 0]) / 2,
                               (person_kpts_np[11, 1] + person_kpts_np[12, 1]) / 2])
            vector = hip_mid - shoulder_mid
            angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi
            angle_deviation = abs(angle - 90)
            if angle_deviation > 60 and (person_box[2] - person_box[0]) > (person_box[3] - person_box[1]):
                wrist_left = person_kpts_np[9]
                wrist_right = person_kpts_np[10]
                shoulder_left = person_kpts_np[5]
                shoulder_right = person_kpts_np[6]
                ankle_left = person_kpts_np[15]
                ankle_right = person_kpts_np[16]
                hip_left = person_kpts_np[11]
                hip_right = person_kpts_np[12]
                if (wrist_left[1] > shoulder_left[1] and wrist_right[1] > shoulder_right[1]) and \
                   (ankle_left[1] > hip_left[1] and ankle_right[1] > hip_right[1]):
                    wrist_distance = np.linalg.norm(wrist_left[:2] - wrist_right[:2])
                    shoulder_distance = np.linalg.norm(shoulder_left[:2] - shoulder_right[:2])
                    if wrist_distance > 1.2 * shoulder_distance:
                        return True
        return False

    def is_climbing(self, person_kpts, person_box, frame_height):
        person_kpts_np = person_kpts.cpu().numpy()

        # Định nghĩa các chỉ số keypoint
        left_eye = 1  # Mắt trái
        right_eye = 2  # Mắt phải
        left_ear = 3  # Tai trái
        right_ear = 4  # Tai phải
        left_wrist, right_wrist = 9, 10  # Cổ tay trái, phải
        left_hip, right_hip = 11, 12  # Hông trái, phải
        left_knee, right_knee = 13, 14  # Đầu gối trái, phải

        # Lấy tọa độ y của các điểm đầu (tai và mắt) với độ tin cậy > 0.5
        head_indices = [left_eye, right_eye, left_ear, right_ear]
        head_points_y = [person_kpts_np[idx, 1] for idx in head_indices if person_kpts_np[idx, 2] > 0.5]
        if len(head_points_y) == 0:
            return False  # Không có điểm đầu nào để so sánh

        # Lấy tọa độ y của cổ tay với độ tin cậy > 0.5
        wrist_y_list = []
        if person_kpts_np[left_wrist, 2] > 0.5:
            wrist_y_list.append(person_kpts_np[left_wrist, 1])
        if person_kpts_np[right_wrist, 2] > 0.5:
            wrist_y_list.append(person_kpts_np[right_wrist, 1])
        if len(wrist_y_list) == 0:
            return False  # Không có cổ tay nào để so sánh

        # Kiểm tra xem có ít nhất một cổ tay cao hơn ít nhất một điểm đầu hay không
        wrist_above_head = any(wrist_y < head_y for wrist_y in wrist_y_list for head_y in head_points_y)

        # Kiểm tra điều kiện đầu gối cao hơn hông
        knee_above_hip = False
        if person_kpts_np[left_knee, 2] > 0.5 and person_kpts_np[left_hip, 2] > 0.5:
            if person_kpts_np[left_knee, 1] < person_kpts_np[left_hip, 1]:
                knee_above_hip = True
        if person_kpts_np[right_knee, 2] > 0.5 and person_kpts_np[right_hip, 2] > 0.5:
            if person_kpts_np[right_knee, 1] < person_kpts_np[right_hip, 1]:
                knee_above_hip = True

        # Kết hợp các điều kiện: cổ tay trên đầu và đầu gối trên hông
        if wrist_above_head:
            return True
        return False

    def initialize_video_writer(self, frame):
        if self.opt.save and self.vid_writer is None:
            fps = 30 if self.source_type == 'webcam' else int(self.cap.get(cv2.CAP_PROP_FPS))
            w, h = frame.shape[1], frame.shape[0]
            output_path = str(self.save_dir / f'output_{int(time.time())}.mp4')
            self.vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            print(f"Đã khởi tạo video writer tại: {output_path}")

    def process_image(self):
        image = cv2.imread(self.opt.source)
        pose_results = self.pose_model.predict(image, conf=self.opt.conf, imgsz=self.opt.imgsz)
        if pose_results and pose_results[0].keypoints is not None:
            kpts = pose_results[0].keypoints
            boxes = pose_results[0].boxes.xyxy if pose_results[0].boxes is not None else None
            frame_height = image.shape[0]
            for i in range(kpts.data.shape[0]):
                person_kpts = kpts.data[i]
                if boxes is not None and i < boxes.shape[0]:
                    person_box = boxes[i].tolist()
                    if self.is_falling(person_kpts, person_box, frame_height):
                        label = "Falling"
                        color = (0, 0, 255)
                    elif self.is_climbing(person_kpts, person_box, frame_height):
                        label = "Climbing"
                        color = (255, 0, 0)
                    elif self.is_crawling(person_kpts, person_box, frame_height):
                        label = "Crawling"
                        color = (0, 255, 255)
                    else:
                        label = "Normal"
                        color = (0, 255, 0)
                    p1 = (int(person_box[0]), int(person_box[1]))
                    p2 = (int(person_box[2]), int(person_box[3]))
                    cv2.rectangle(image, p1, p2, color, 2)
                    cv2.putText(image, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            self.draw_skeleton(image, kpts)
        if self.opt.view:
            cv2.imshow("Pose Detection", image)
            cv2.waitKey(0)
        if self.opt.save:
            output_path = str(self.save_dir / Path(self.opt.source).name)
            cv2.imwrite(output_path, image)
            print(f"Ảnh đã lưu tại: {output_path}")

    def process_video_stream(self):
        if self.opt.view:
            cv2.namedWindow("Real-time Pose Detection", cv2.WINDOW_NORMAL)
        while self.running:
            if self.source_type in ['stream', 'webcam'] and self.frame_queue is not None:
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
            frame_height = frame.shape[0]
            pose_results = self.pose_model.predict(frame, conf=self.opt.conf, imgsz=self.opt.imgsz)
            if pose_results and pose_results[0].keypoints is not None:
                kpts = pose_results[0].keypoints
                boxes = pose_results[0].boxes.xyxy if pose_results[0].boxes is not None else None
                for i in range(kpts.data.shape[0]):
                    person_kpts = kpts.data[i]
                    if boxes is not None and i < boxes.shape[0]:
                        person_box = boxes[i].tolist()
                        if self.is_falling(person_kpts, person_box, frame_height):
                            label = "Falling"
                            color = (0, 0, 255)
                        elif self.is_climbing(person_kpts, person_box, frame_height):
                            label = "Climbing"
                            color = (255, 0, 0)
                        elif self.is_crawling(person_kpts, person_box, frame_height):
                            label = "Crawling"
                            color = (0, 255, 255)
                        else:
                            label = "Normal"
                            color = (0, 255, 0)
                        p1 = (int(person_box[0]), int(person_box[1]))
                        p2 = (int(person_box[2]), int(person_box[3]))
                        cv2.rectangle(frame, p1, p2, color, 2)
                        cv2.putText(frame, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                self.draw_skeleton(frame, kpts)
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
        """Đọc frame từ nguồn video/stream và đưa vào queue, xử lý lỗi OpenCV"""
        while self.running:
            if self.source_type == 'stream' and self.opt.source.startswith('rtsp://'):
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.opt.source)
                    if not self.cap.isOpened():
                        time.sleep(5)
                        continue
            try:
                ret, frame = self.cap.read()
                if not ret:
                    if self.source_type == 'stream':
                        self.cap.release()
                        time.sleep(5)
                        continue
                    else:
                        break
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(frame)
            except cv2.error as e:
                print(f"OpenCV error: {e}")
                if self.source_type == 'stream':
                    self.cap.release()
                    time.sleep(5)
                    continue
                else:
                    break

    def run(self):
        if self.source_type == 'image':
            self.process_image()
        else:
            if self.source_type in ['stream', 'webcam']:
                self.reader_thread = threading.Thread(target=self.frame_reader)
                self.reader_thread.start()
            self.process_video_stream()

def parse_args():
    parser = argparse.ArgumentParser(description='Pose Detection System')
    parser.add_argument('--weights', type=str, default='yolo11m-pose.pt', help='Đường dẫn mô hình pose')
    parser.add_argument('--source', type=str, default='0', help='Ảnh/Video/Webcam/Stream URL')
    parser.add_argument('--imgsz', type=int, default=640, help='Kích thước inference')
    parser.add_argument('--conf', type=float, default=0.4, help='Ngưỡng confidence')
    parser.add_argument('--view', action='store_true', help='Hiển thị kết quả real-time')
    parser.add_argument('--save', action='store_true', help='Lưu kết quả đầu ra liên tục')
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_args()
    detector = PoseDetectionSystem(opt)
    detector.run()