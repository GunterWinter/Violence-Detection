import cv2
import time
import math
import numpy as np
import threading
import queue
from pathlib import Path
from datetime import datetime

from sympy import false
from ultralytics import YOLO
from ffmpeg import FFmpeg

class ViolencePoseDetectionSystem:
    def __init__(self, opt):
        """Khởi tạo hệ thống phát hiện bạo lực và tư thế.

        Args:
            opt (dict): Tùy chọn cấu hình bao gồm đường dẫn mô hình, nguồn, v.v.
        """
        self.opt = opt
        self.violence_model = YOLO(opt['violence_weights'])
        self.pose_model = YOLO(opt['weights'])
        self.output_dir = Path(opt['record_dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.source_type = self.determine_source_type()
        self.cap = None
        self.vid_writer = None
        self.save = opt['save']
        self.tail_length = opt['tail_length']
        self.recording = False
        self.ffmpeg_process = None
        self.activity_count = 0
        self.tail_frames = None
        self.running = True
        self.use_thread = self.source_type in ['stream', 'webcam']
        self.frame_queue = queue.Queue(maxsize=10) if self.use_thread else None
        self.output_path = None
        self.is_recording = False
        self.ffmpeg_thread = None

        if self.source_type != 'image':
            self.cap = cv2.VideoCapture(0 if self.source_type == 'webcam' else self.opt['source'])
            if not self.cap.isOpened():
                raise ValueError("Không thể mở nguồn video/stream")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
            self.tail_frames = self.tail_length * self.fps

    def determine_source_type(self):
        """Xác định loại nguồn đầu vào.

        Returns:
            str: Loại nguồn ('image', 'video', 'webcam', 'stream').
        """
        source = self.opt['source']
        if source.isnumeric():
            return 'webcam'
        if source.lower().startswith(('http://', 'https://', 'rtsp://')):
            return 'stream'
        if Path(source).exists():
            if Path(source).suffix.lower() in ['.jpg', '.png', '.jpeg']:
                return 'image'
            return 'video'
        raise ValueError("Nguồn đầu vào không hợp lệ")

    def frame_reader(self):
        """Đọc frame từ nguồn video/stream và đưa vào queue."""
        while self.running:
            if self.source_type == 'stream' and self.opt['source'].startswith('rtsp://'):
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.opt['source'])
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
        """Cắt ảnh theo bounding box.

        Args:
            image (np.ndarray): Ảnh đầu vào.
            box: Bounding box từ kết quả YOLO.

        Returns:
            tuple: Ảnh đã cắt và offset (xmin, ymin).
        """
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
        return image[ymin:ymax, xmin:xmax], (xmin, ymin)

    def draw_skeleton(self, image, person_kpts, offset):
        """Vẽ bộ khung xương cho một người.

        Args:
            image (np.ndarray): Ảnh để vẽ lên.
            person_kpts: Keypoints của một người.
            offset (tuple): Offset (x, y) để điều chỉnh tọa độ.
        """
        skeleton_pairs = [
            (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        for pair in skeleton_pairs:
            if pair[0] < person_kpts.shape[0] and pair[1] < person_kpts.shape[0]:
                pt1, pt2 = person_kpts[pair[0]], person_kpts[pair[1]]
                if pt1[2] > 0.5 and pt2[2] > 0.5:
                    x1, y1 = int(pt1[0] + offset[0]), int(pt1[1] + offset[1])
                    x2, y2 = int(pt2[0] + offset[0]), int(pt2[1] + offset[1])
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for kpt in person_kpts:
            if kpt[2] > 0.5:
                x, y = int(kpt[0] + offset[0]), int(kpt[1] + offset[1])
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    def is_falling(self, person_kpts, person_box, frame_height):
        """Kiểm tra xem người có đang ngã hay không dựa trên góc nghiêng.

        Args:
            person_kpts: Keypoints của một người.
            person_box: Bounding box của người.
            frame_height (int): Chiều cao khung hình.

        Returns:
            bool: True nếu người đang ngã.
        """
        required_keypoints = [5, 6, 11, 12]
        for idx in required_keypoints:
            if person_kpts[idx][2] <= 0.5:
                return False

        left_shoulder, right_shoulder = person_kpts[5], person_kpts[6]
        left_hip, right_hip = person_kpts[11], person_kpts[12]

        def get_vector(pt1, pt2):
            if pt1[2] > 0.5 and pt2[2] > 0.5:
                return pt2[0] - pt1[0], pt2[1] - pt1[1]
            return None

        left_vector = get_vector(left_shoulder, left_hip)
        right_vector = get_vector(right_shoulder, right_hip)

        if left_vector is None and right_vector is None:
            return False

        def get_angle(vector):
            if vector:
                dx, dy = vector
                angle = math.degrees(math.atan2(dy, dx))
                return angle
            return None

        left_angle = get_angle(left_vector)
        right_angle = get_angle(right_vector)

        if left_angle is not None and right_angle is not None:
            avg_angle = (left_angle + right_angle) / 2
        elif left_angle is not None:
            avg_angle = left_angle
        elif right_angle is not None:
            avg_angle = right_angle
        else:
            return False

        deviation = abs(avg_angle - 90)
        return deviation > 45

    def initialize_video_writer(self, frame):
        """Khởi tạo video writer để ghi video đầu ra.

        Args:
            frame (np.ndarray): Frame đầu tiên để lấy kích thước.
        """
        if self.vid_writer is None:
            if self.source_type == 'video':
                fps = self.cap.get(cv2.CAP_PROP_FPS)
            else:
                fps = 30  # Mặc định cho webcam
            w, h = frame.shape[1], frame.shape[0]
            output_path = str(self.output_dir / f'output_{int(time.time())}.mp4')
            self.vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.output_path = output_path
            print(f"Đã khởi tạo video writer tại: {output_path} với FPS: {fps}")

    def start_recording(self):
        """Bắt đầu ghi hình từ RTSP stream bằng FFmpeg."""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = self.output_dir / f'recording_{timestamp}.mkv'
        self.ffmpeg_process = (
            FFmpeg()
            .option("y")
            .input(self.opt['source'], rtsp_transport="tcp", rtsp_flags="prefer_tcp")
            .output(str(filename), vcodec="copy", acodec="copy")
        )
        self.ffmpeg_thread = threading.Thread(target=self.ffmpeg_process.execute)
        self.ffmpeg_thread.start()
        self.recording = True
        print(f"Bắt đầu ghi hình: {filename}")

    def stop_recording(self):
        """Dừng ghi hình từ RTSP stream."""
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.ffmpeg_thread.join()
            self.ffmpeg_process = None
            self.recording = False
            print("Đã dừng ghi hình")

    def start_screen_recording(self):
        """Bắt đầu quay màn hình bằng FFmpeg."""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = self.output_dir / f'screen_recording_{timestamp}.mkv'
        self.ffmpeg_process = (
            FFmpeg()
            .option("y")
            .input(self.opt['source'], rtsp_transport="tcp" if self.source_type == 'stream' else None)
            .output(str(filename), vcodec="copy", acodec="copy")
        )
        self.ffmpeg_thread = threading.Thread(target=self.ffmpeg_process.execute)
        self.ffmpeg_thread.start()
        self.is_recording = True
        print(f"Bắt đầu quay màn hình: {filename}")

    def stop_screen_recording(self):
        """Dừng quay màn hình."""
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.ffmpeg_thread.join()
            self.ffmpeg_process = None
            self.is_recording = False
            print("Đã dừng quay màn hình")

    def _process_frame(self, frame):
        """Xử lý một frame để phát hiện bạo lực và tư thế.

        Args:
            frame (np.ndarray): Frame đầu vào.

        Returns:
            tuple: Frame đã xử lý và boolean chỉ ra có phát hiện bạo lực hay không.
        """
        frame_height = frame.shape[0]
        violence_results = self.violence_model.predict(frame, conf=self.opt['conf'], imgsz=self.opt['imgsz'],
                                                       verbose=False)
        violence_class_id = next(k for k, v in self.violence_model.names.items() if v == 'Violence')
        violence_boxes = [box.xyxy[0].tolist() for box in violence_results[0].boxes if box.cls == violence_class_id]
        violence_detected = len(violence_boxes) > 0
        falling = False
        pose_results = self.pose_model.predict(frame, conf=self.opt['conf'], imgsz=self.opt['imgsz'], verbose=False)
        if pose_results and pose_results[0].keypoints is not None and pose_results[0].boxes is not None:
            kpts = pose_results[0].keypoints.data
            boxes = pose_results[0].boxes.xyxy
            for i in range(boxes.shape[0]):
                person_box = boxes[i].tolist()
                for violence_box in violence_boxes:
                    if self.is_box_intersecting(person_box, violence_box):
                        person_kpts = kpts[i]
                        if self.is_falling(person_kpts, person_box, frame_height):
                            falling = True
                            xmin, ymin, xmax, ymax = map(int, person_box)
                            cv2.putText(frame, "Falling", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255),
                                        2)
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                        self.draw_skeleton(frame, person_kpts, (0, 0))
                        break

        for box in violence_boxes:
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[2]), int(box[3]))
            if(falling):
                label, color = "Serious Violence", (0, 0, 255)
            else:
                label, color = "Violence", (0, 255, 0)
            cv2.rectangle(frame, p1, p2, color, 2)
            cv2.putText(frame, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return frame, violence_detected

    def is_box_intersecting(self, box1, box2):
        """Kiểm tra xem hai bounding box có giao nhau hay không.

        Args:
            box1 (list): [xmin, ymin, xmax, ymax] của box thứ nhất.
            box2 (list): [xmin, ymin, xmax, ymax] của box thứ hai.

        Returns:
            bool: True nếu hai box giao nhau.
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

    def process_image(self):
        """Xử lý ảnh đầu vào và lưu kết quả nếu cần."""
        image = cv2.imread(self.opt['source'])
        image, _ = self._process_frame(image)
        if self.opt['view']:
            cv2.imshow("Violence Detection", image)
            cv2.waitKey(0)
        if self.save:
            output_path = str(self.output_dir / Path(self.opt['source']).name)
            cv2.imwrite(output_path, image)
            self.output_path = output_path
            print(f"Ảnh đã lưu tại: {output_path}")

    def process_video_directly(self):
        """Xử lý video trực tiếp từ file video với multi-threading."""
        self.cap = cv2.VideoCapture(self.opt['source'])
        if not self.cap.isOpened():
            raise ValueError("Không thể mở video")
        if self.opt.get('view', False):
            cv2.namedWindow("Violence Detection", cv2.WINDOW_NORMAL)

        frame_queue = queue.Queue(maxsize=100)

        # Hàm đọc frame chạy trong thread riêng
        def frame_reader():
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    break
                # Đợi đến khi queue có chỗ trống
                while frame_queue.full():
                    time.sleep(0.01)
                frame_queue.put(frame)
            frame_queue.put(None)  # Đánh dấu kết thúc

        # Khởi động thread đọc frame
        reader_thread = threading.Thread(target=frame_reader)
        reader_thread.start()

        # Xử lý và ghi frame
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            processed_frame, _ = self._process_frame(frame)
            if self.save:
                if self.vid_writer is None:
                    self.initialize_video_writer(processed_frame)
                self.vid_writer.write(processed_frame)
            if self.opt.get('view', False):
                cv2.imshow("Violence Detection", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Dọn dẹp
        reader_thread.join()
        if self.vid_writer:
            self.vid_writer.release()
        self.cap.release()
        cv2.destroyAllWindows()

    def process_video_stream(self):
        """Xử lý video stream hoặc webcam trong luồng riêng."""
        if self.opt['view']:
            cv2.namedWindow("Real-time Violence Detection", cv2.WINDOW_NORMAL)
        self.reader_thread = threading.Thread(target=self.frame_reader)
        self.reader_thread.start()

        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                if frame is None:
                    break
                processed_frame, violence_detected = self._process_frame(frame)
                if self.save:
                    if self.source_type == 'stream' and self.opt['source'].startswith('rtsp://'):
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
                            self.initialize_video_writer(processed_frame)
                        self.vid_writer.write(processed_frame)
                if self.opt['view']:
                    cv2.imshow("Real-time Violence Detection", processed_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.running = False
                        break
                    elif key == ord('r'):
                        if not self.is_recording:
                            self.start_screen_recording()
                        else:
                            self.stop_screen_recording()
            else:
                time.sleep(0.01)

        if self.vid_writer:
            self.vid_writer.release()
        if self.recording:
            self.stop_recording()
        if self.is_recording:
            self.stop_screen_recording()
        self.running = False
        self.reader_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        """Lấy frame từ queue để stream.

        Returns:
            bytes: Frame mã hóa dưới dạng JPEG hoặc None.
        """
        if self.frame_queue and not self.frame_queue.empty():
            frame = self.frame_queue.get()
            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes()
        return None

    def run(self):
        """Chạy hệ thống phát hiện dựa trên loại nguồn đầu vào."""
        if self.source_type == 'image':
            self.process_image()
        elif self.use_thread:
            self.process_video_stream()
        else:
            self.process_video_directly()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Violence and Pose Detection System')
    parser.add_argument('--weights', type=str, default='yolo11n-pose.pt', help='Đường dẫn mô hình pose')
    parser.add_argument('--violence-weights', type=str,
                        default=r'C:\BaiTap\Python\Violence_Detection\Yolo11_Violence_Detection\runs\detect\train\weights\best.pt',
                        help='Đường dẫn mô hình Violence')
    parser.add_argument('--source', type=str, default='0', help='Ảnh/Video/Webcam/Stream URL')
    parser.add_argument('--imgsz', type=int, default=640, help='Kích thước inference')
    parser.add_argument('--conf', type=float, default=0.4, help='Ngưỡng confidence')
    parser.add_argument('--view', action='store_true', help='Hiển thị kết quả real-time')
    parser.add_argument('--save', action='store_true', help='Lưu kết quả hoặc ghi hình nếu là RTSP stream')
    parser.add_argument('--tail_length', type=int, default=5,
                        help='Thời gian (giây) tiếp tục ghi sau khi không phát hiện bạo lực')
    parser.add_argument('--record_dir', type=str, default='recordings', help='Thư mục lưu trữ kết quả')
    args = parser.parse_args()

    opt = vars(args)
    detector = ViolencePoseDetectionSystem(opt)
    detector.run()