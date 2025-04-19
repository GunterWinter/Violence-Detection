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
        # Khởi tạo các tham số và mô hình
        self.opt = opt
        self.violence_model = YOLO(opt.violence_weights)
        self.pose_model = YOLO(opt.weights)  # Tải mô hình YOLO để ước lượng tư thế
        self.save_dir = Path('results')  # Thư mục lưu kết quả đầu ra
        self.save_dir.mkdir(exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
        self.record_dir = Path('recordings')  # Thư mục lưu video ghi lại từ luồng RTSP
        self.record_dir.mkdir(exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
        self.source_type = self.determine_source_type()  # Xác định loại nguồn đầu vào
        self.cap = None  # Đối tượng video capture
        self.vid_writer = None  # Đối tượng ghi video kết quả
        self.save = opt.save  # Cờ bật/tắt lưu kết quả hoặc ghi hình
        self.tail_length = opt.tail_length  # Thời gian ghi thêm sau khi không còn bạo lực (giây)
        self.recording = False  # Trạng thái ghi hình
        self.ffmpeg_process = None  # Tiến trình FFmpeg để ghi hình từ RTSP
        self.activity_count = 0  # Đếm số khung hình không có bạo lực để dừng ghi
        self.tail_frames = None  # Số khung hình tương ứng với tail_length
        self.frame_queue = queue.Queue(maxsize=10)  # Hàng đợi để lưu khung hình
        self.running = True  # Cờ để kiểm soát vòng lặp chính

        # Nếu không phải ảnh, khởi tạo video capture và luồng đọc khung hình
        if self.source_type != 'image':
            self.cap = cv2.VideoCapture(0 if self.source_type == 'webcam' else self.opt.source)
            if not self.cap.isOpened():
                raise ValueError("Không thể mở nguồn video/stream")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25  # Lấy FPS, mặc định 25 nếu không xác định được
            self.tail_frames = self.tail_length * self.fps  # Tính số khung hình cho tail_length
            self.reader_thread = threading.Thread(target=self.frame_reader)  # Tạo luồng đọc khung hình
            self.reader_thread.start()  # Bắt đầu luồng

    def determine_source_type(self):
        # Xác định loại nguồn đầu vào dựa trên tham số source
        if self.opt.source.isnumeric():
            return 'webcam'  # Nguồn là webcam nếu là số (e.g., 0)
        if self.opt.source.lower().startswith(('http://', 'https://', 'rtsp://')):
            return 'stream'  # Nguồn là luồng nếu bắt đầu bằng http/https/rtsp
        if Path(self.opt.source).exists():
            if Path(self.opt.source).suffix.lower() in ['.jpg', '.png', '.jpeg']:
                return 'image'  # Nguồn là ảnh nếu có đuôi ảnh
            return 'video'  # Nguồn là video nếu không phải ảnh
        raise ValueError("Nguồn đầu vào không hợp lệ")

    def frame_reader(self):
        # Luồng đọc khung hình từ nguồn video/stream và đưa vào hàng đợi
        while self.running:
            if self.source_type == 'stream' and self.opt.source.startswith('rtsp://'):
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.opt.source)  # Thử mở lại nếu mất kết nối
                    if not self.cap.isOpened():
                        time.sleep(5)  # Chờ 5 giây trước khi thử lại
                        continue
            ret, frame = self.cap.read()  # Đọc khung hình
            if ret:
                if self.frame_queue.full():
                    self.frame_queue.get()  # Xóa khung cũ nếu hàng đợi đầy
                self.frame_queue.put(frame)  # Đưa khung hình vào hàng đợi
            else:
                if self.source_type == 'video':
                    self.frame_queue.put(None)  # Đánh dấu kết thúc video
                    break
                elif self.source_type == 'stream':
                    self.cap.release()  # Giải phóng capture nếu mất kết nối
                    time.sleep(5)  # Chờ trước khi thử lại
                else:
                    time.sleep(0.1)  # Nghỉ ngắn nếu không có khung hình

    def crop_image(self, image, box):
        # Cắt vùng ảnh chứa đối tượng từ bounding box
        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()  # Lấy tọa độ bounding box
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)  # Chuyển sang số nguyên
        return image[ymin:ymax, xmin:xmax], (xmin, ymin)  # Trả về ảnh cắt và offset

    def draw_skeleton(self, image, kpts, offset):
        # Vẽ khung xương của người lên ảnh
        if kpts is None or len(kpts.data) == 0:
            return  # Không vẽ nếu không có keypoints
        for person_kpts in kpts.data:
            for kpt in person_kpts:
                if kpt[2] > 0:  # Nếu keypoint có độ tin cậy cao
                    x, y = int(kpt[0] + offset[0]), int(kpt[1] + offset[1])  # Tính tọa độ thực tế
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Vẽ điểm tròn

    def is_falling(self, person_kpts, person_box, frame_height):
        # Kiểm tra xem người có đang té ngã dựa trên vị trí vai và bounding box
        left_shoulder_y = person_kpts[5][1] if person_kpts[5][2] > 0 else None  # Tọa độ y của vai trái
        right_shoulder_y = person_kpts[6][1] if person_kpts[6][2] > 0 else None  # Tọa độ y của vai phải
        shoulder_y = None
        if left_shoulder_y and right_shoulder_y:
            shoulder_y = (left_shoulder_y + right_shoulder_y) / 2  # Trung bình tọa độ vai
        elif left_shoulder_y:
            shoulder_y = left_shoulder_y
        elif right_shoulder_y:
            shoulder_y = right_shoulder_y
        if not shoulder_y:
            return False  # Không xác định được vai
        xmin, ymin, xmax, ymax = person_box
        dx, dy = xmax - xmin, ymax - ymin  # Tính chiều rộng và cao của box
        difference = dy - dx  # So sánh chiều cao và chiều rộng
        thre = (frame_height // 2) + 100  # Ngưỡng để xác định té ngã
        return (difference <= 0 and shoulder_y > thre) or (difference < 0)  # Điều kiện té ngã

    def initialize_video_writer(self, frame):
        # Khởi tạo đối tượng ghi video nếu cần lưu kết quả
        if self.vid_writer is None:
            fps = 30 if self.source_type == 'webcam' else int(self.cap.get(cv2.CAP_PROP_FPS))  # Xác định FPS
            w, h = frame.shape[1], frame.shape[0]  # Lấy kích thước khung hình
            output_path = str(self.save_dir / f'output_{int(time.time())}.mp4')  # Đường dẫn file đầu ra
            self.vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            print(f"Đã khởi tạo video writer tại: {output_path}")

    def start_recording(self):
        # Bắt đầu ghi video gốc từ luồng RTSP khi phát hiện bạo lực
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Tạo tên file theo thời gian
        filename = self.record_dir / f'recording_{timestamp}.mkv'  # Đường dẫn file ghi hình
        self.ffmpeg_process = (
            FFmpeg()
            .option("y")  # Ghi đè nếu file đã tồn tại
            .input(self.opt.source, rtsp_transport="tcp", rtsp_flags="prefer_tcp")  # Nguồn RTSP
            .output(str(filename), vcodec="copy", acodec="copy")  # Ghi hình không mã hóa lại
        )
        self.ffmpeg_thread = threading.Thread(target=self.ffmpeg_process.execute)  # Tạo luồng FFmpeg
        self.ffmpeg_thread.start()  # Bắt đầu ghi
        self.recording = True  # Cập nhật trạng thái
        print(f"Bắt đầu ghi hình: {filename}")

    def stop_recording(self):
        # Dừng ghi hình khi không còn bạo lực trong khoảng thời gian tail_length
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()  # Dừng tiến trình FFmpeg
            self.ffmpeg_thread.join()  # Chờ luồng kết thúc
            self.ffmpeg_process = None
            self.recording = False  # Cập nhật trạng thái
            print("Đã dừng ghi hình")

    def process_image(self):
        # Xử lý ảnh tĩnh: phát hiện bạo lực và vẽ kết quả
        image = cv2.imread(self.opt.source)  # Đọc ảnh từ file
        violence_results = self.violence_model.predict(image, conf=self.opt.conf, imgsz=self.opt.imgsz, verbose=False)  # Dự đoán bạo lực
        violence_class_id = next(k for k, v in self.violence_model.names.items() if v == 'Violence')  # Lấy ID lớp Violence
        violence_boxes = [box for box in violence_results[0].boxes if box.cls == violence_class_id]  # Lọc bounding box bạo lực

        for box in violence_boxes:
            cropped_image, offset = self.crop_image(image, box)  # Cắt vùng chứa bạo lực
            pose_results = self.pose_model.predict(cropped_image, conf=self.opt.conf, imgsz=self.opt.imgsz, verbose=False)  # Dự đoán tư thế
            p1 = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))  # Tọa độ góc trên trái
            p2 = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))  # Tọa độ góc dưới phải
            cv2.rectangle(image, p1, p2, (0, 255, 0), 2)  # Vẽ khung bạo lực
            cv2.putText(image, "Violence", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Ghi nhãn
            if pose_results and pose_results[0].keypoints is not None:
                self.draw_skeleton(image, pose_results[0].keypoints, offset)  # Vẽ khung xương

        if self.opt.view:
            cv2.imshow("Violence Detection", image)  # Hiển thị kết quả
            cv2.waitKey(0)  # Chờ người dùng đóng cửa sổ
        if self.save:
            output_path = str(self.save_dir / Path(self.opt.source).name)  # Đường dẫn lưu ảnh
            cv2.imwrite(output_path, image)  # Lưu ảnh
            print(f"Ảnh đã lưu tại: {output_path}")

    def process_video_stream(self):
        # Xử lý video hoặc luồng thời gian thực
        if self.opt.view:
            cv2.namedWindow("Real-time Violence Detection", cv2.WINDOW_NORMAL)  # Tạo cửa sổ có thể thay đổi kích thước

        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()  # Lấy khung hình từ hàng đợi
                if frame is None:
                    break  # Thoát nếu hết video
                frame_height = frame.shape[0]  # Lấy chiều cao khung hình
                violence_results = self.violence_model.predict(frame, conf=self.opt.conf, imgsz=self.opt.imgsz, verbose=False)  # Dự đoán bạo lực
                violence_class_id = next(k for k, v in self.violence_model.names.items() if v == 'Violence')  # Lấy ID lớp Violence
                violence_boxes = [box for box in violence_results[0].boxes if box.cls == violence_class_id]  # Lọc box bạo lực
                violence_detected = len(violence_boxes) > 0  # Kiểm tra có bạo lực hay không

                # Xử lý từng vùng phát hiện bạo lực
                for box in violence_boxes:
                    cropped_frame, offset = self.crop_image(frame, box)  # Cắt vùng bạo lực
                    pose_results = self.pose_model.predict(cropped_frame, conf=self.opt.conf, imgsz=self.opt.imgsz, verbose=False)  # Dự đoán tư thế
                    p1 = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))  # Tọa độ góc trên trái
                    p2 = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))  # Tọa độ góc dưới phải
                    label = "Violence"
                    color = (0, 255, 0)  # Màu mặc định cho bạo lực
                    if pose_results and pose_results[0].keypoints is not None and pose_results[0].boxes is not None:
                        kpts = pose_results[0].keypoints.data  # Lấy keypoints
                        boxes = pose_results[0].boxes.xyxy  # Lấy bounding boxes của người
                        if boxes.shape[0] > 0:
                            for i in range(kpts.shape[0]):
                                if i < boxes.shape[0]:
                                    person_kpts = kpts[i]  # Keypoints của từng người
                                    person_box = boxes[i].clone()  # Bounding box của từng người
                                    person_box[0] += offset[0]  # Điều chỉnh tọa độ thực tế
                                    person_box[1] += offset[1]
                                    person_box[2] += offset[0]
                                    person_box[3] += offset[1]
                                    if self.is_falling(person_kpts, person_box, frame_height):  # Kiểm tra té ngã
                                        xmin, ymin, xmax, ymax = person_box.int().tolist()
                                        cv2.putText(frame, "Falling", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                                    (0, 0, 255), 2)  # Ghi nhãn té ngã
                                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # Vẽ khung đỏ
                                        label = "Serious Violence"
                                        color = (0, 0, 255)
                    cv2.rectangle(frame, p1, p2, color, 2)
                    cv2.putText(frame, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Ghi nhãn
                    if pose_results and pose_results[0].keypoints is not None:
                        self.draw_skeleton(frame, pose_results[0].keypoints, offset)  # Vẽ khung xương

                # Ghi hình nếu là RTSP stream và --save được kích hoạt
                if self.save and self.source_type == 'stream' and self.opt.source.startswith('rtsp://'):
                    if violence_detected:
                        if not self.recording:
                            self.start_recording()  # Bắt đầu ghi hình
                        self.activity_count = 0  # Đặt lại bộ đếm
                    elif self.recording:
                        self.activity_count += 1  # Tăng bộ đếm khi không có bạo lực
                        if self.activity_count > self.tail_frames:
                            self.stop_recording()  # Dừng ghi hình sau tail_length

                # Lưu video kết quả nếu --save được kích hoạt và không phải RTSP stream
                if self.save and self.source_type != 'stream':
                    self.initialize_video_writer(frame)  # Khởi tạo video writer
                    if self.vid_writer is not None:
                        self.vid_writer.write(frame)  # Ghi khung hình

                if self.opt.view:
                    cv2.imshow("Real-time Violence Detection", frame)  # Hiển thị khung hình

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False  # Thoát nếu nhấn 'q'
                    break
            else:
                time.sleep(0.01)  # Nghỉ ngắn nếu không có khung hình

        # Giải phóng tài nguyên khi kết thúc
        if self.vid_writer:
            self.vid_writer.release()  # Giải phóng video writer
        if self.recording:
            self.stop_recording()
        self.running = False
        if hasattr(self, 'reader_thread'):
            self.reader_thread.join()  # Chờ luồng đọc khung hình kết thúc
        self.cap.release()  # Giải phóng video capture
        cv2.destroyAllWindows()  # Đóng tất cả cửa sổ

    def run(self):
        # Chạy hệ thống dựa trên loại nguồn đầu vào
        if self.source_type == 'image':
            self.process_image()  # Xử lý ảnh tĩnh
        else:
            self.process_video_stream()  # Xử lý video/luồng


def parse_args():
    # Phân tích các tham số dòng lệnh
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
    detector = ViolencePoseDetectionSystem(opt)  #
    detector.run()