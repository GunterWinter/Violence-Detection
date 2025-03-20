import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import torch
import numpy as np


class ViolencePoseDetectionSystem:
    def __init__(self, opt):
        self.opt = opt
        self.violence_model = YOLO(opt.violence_weights)  # Mô hình phát hiện Violence
        self.pose_model = YOLO(opt.weights)  # Mô hình pose estimation
        self.save_dir = Path('results')
        self.save_dir.mkdir(exist_ok=True)
        self.source_type = self.determine_source_type()
        self.cap = None
        self.vid_writer = None

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

    def crop_image(self, image, box):
        """Cắt ảnh trong bounding box của Violence"""
        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        return image[ymin:ymax, xmin:xmax], (xmin, ymin)

    def draw_skeleton(self, image, kpts, offset):
        """Vẽ khung xương lên ảnh gốc với offset của bounding box"""
        if kpts is None or len(kpts.data) == 0:
            return  # Thoát nếu không có keypoint
        keypoints_data = kpts.data
        for person_kpts in keypoints_data:
            for kpt in person_kpts:
                if kpt[2] > 0:
                    x, y = int(kpt[0] + offset[0]), int(kpt[1] + offset[1])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    def is_falling(self, person_kpts, person_box, frame_height):
        """Kiểm tra xem một người có đang té ngã hay không"""
        # Lấy tọa độ y của vai trái (5) và vai phải (6)
        left_shoulder_y = person_kpts[5][1] if person_kpts[5][2] > 0 else None
        right_shoulder_y = person_kpts[6][1] if person_kpts[6][2] > 0 else None

        # Tính trung bình tọa độ y của vai
        if left_shoulder_y and right_shoulder_y:
            shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        elif left_shoulder_y:
            shoulder_y = left_shoulder_y
        elif right_shoulder_y:
            shoulder_y = right_shoulder_y
        else:
            return False  # Không phát hiện vai

        # Tính difference
        xmin, ymin, xmax, ymax = person_box
        dx = xmax - xmin
        dy = ymax - ymin
        difference = dy - dx
        thre = (frame_height // 2) + 100

        # Điều kiện té ngã
        if (difference <= 0 and shoulder_y > thre) or (difference < 0):
            return True
        return False

    def initialize_video_writer(self, frame):
        """Khởi tạo VideoWriter để lưu video"""
        if self.opt.save_img and self.vid_writer is None:
            fps = 30 if self.source_type == 'webcam' else int(self.cap.get(cv2.CAP_PROP_FPS))
            w, h = frame.shape[1], frame.shape[0]
            output_path = str(self.save_dir / f'output_{int(time.time())}.mp4')
            self.vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            print(f"Đã khởi tạo video writer tại: {output_path}")

    def process_image(self):
        image = cv2.imread(self.opt.source)
        violence_results = self.violence_model.predict(image, conf=self.opt.conf, imgsz=self.opt.imgsz)
        violence_class_id = next(k for k, v in self.violence_model.names.items() if v == 'violence')
        violence_boxes = [box for box in violence_results[0].boxes if box.cls == violence_class_id]

        for box in violence_boxes:
            cropped_image, offset = self.crop_image(image, box)
            pose_results = self.pose_model.predict(cropped_image, conf=self.opt.conf, imgsz=self.opt.imgsz)
            p1 = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
            p2 = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
            cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
            cv2.putText(image, "Violence", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if pose_results and pose_results[0].keypoints is not None:
                self.draw_skeleton(image, pose_results[0].keypoints, offset)

        if self.opt.view_img:
            cv2.imshow("Violence Detection", image)
            cv2.waitKey(0)
        if self.opt.save_img:
            output_path = str(self.save_dir / Path(self.opt.source).name)
            cv2.imwrite(output_path, image)
            print(f"Ảnh đã lưu tại: {output_path}")

    def process_video_stream(self):
        self.cap = cv2.VideoCapture(0 if self.source_type == 'webcam' else self.opt.source)
        if not self.cap.isOpened():
            raise ValueError("Không thể mở nguồn video/webcam")

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Không thể đọc khung hình, thoát...")
                break

            frame_height = frame.shape[0]  # Lấy chiều cao của frame

            # Phát hiện hành vi bạo lực
            violence_results = self.violence_model.predict(frame, conf=self.opt.conf, imgsz=self.opt.imgsz)
            violence_class_id = next(k for k, v in self.violence_model.names.items() if v == 'Violence')
            violence_boxes = [box for box in violence_results[0].boxes if box.cls == violence_class_id]

            for box in violence_boxes:
                cropped_frame, offset = self.crop_image(frame, box)
                pose_results = self.pose_model.predict(cropped_frame, conf=self.opt.conf, imgsz=self.opt.imgsz)

                p1 = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
                p2 = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))

                label = "Violence"
                color = (0, 255, 0)  # Màu xanh cho Violence

                # Kiểm tra té ngã cho từng người
                if pose_results and pose_results[0].keypoints is not None and pose_results[0].boxes is not None:
                    kpts = pose_results[0].keypoints.data  # (num_persons, 17, 3)
                    boxes = pose_results[0].boxes.xyxy  # (num_persons, 4)
                    if boxes.shape[0] > 0:  # Kiểm tra xem boxes có phần tử nào không
                        for i in range(kpts.shape[0]):
                            if i < boxes.shape[0]:  # Đảm bảo i không vượt quá số lượng boxes
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

                # Vẽ bounding box và nhãn cho Violence
                cv2.rectangle(frame, p1, p2, color, 2)
                cv2.putText(frame, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                if pose_results and pose_results[0].keypoints is not None:
                    self.draw_skeleton(frame, pose_results[0].keypoints, offset)

            if self.opt.view_img:
                cv2.imshow("Real-time Violence Detection", frame)
            if self.opt.save_img:
                self.initialize_video_writer(frame)
                if self.vid_writer is not None:
                    self.vid_writer.write(frame)
                else:
                    print("Lỗi: VideoWriter chưa được khởi tạo!")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if self.vid_writer:
            self.vid_writer.release()
            print("Đã lưu video và giải phóng VideoWriter")
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        if self.source_type == 'image':
            self.process_image()
        else:
            self.process_video_stream()


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
    parser.add_argument('--view-img', action='store_true', help='Hiển thị kết quả real-time')
    parser.add_argument('--save-img', action='store_true', help='Lưu kết quả đầu ra')
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()
    detector = ViolencePoseDetectionSystem(opt)
    detector.run()