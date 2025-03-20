import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import torch
import numpy as np


class ViolencePoseDetectionSystem:
    def __init__(self, opt):
        # Nhận các tham số từ dòng lệnh (opt) để cấu hình hệ thống
        self.opt = opt
        # Tải mô hình YOLO để phát hiện bạo lực từ đường dẫn opt.violence_weights
        self.violence_model = YOLO(opt.violence_weights)
        # Tải mô hình YOLO để ước lượng tư thế từ đường dẫn opt.weights
        self.pose_model = YOLO(opt.weights)
        # Tạo thư mục 'results' để lưu kết quả đầu ra, nếu đã tồn tại thì bỏ qua
        self.save_dir = Path('results')
        self.save_dir.mkdir(exist_ok=True)
        # Xác định loại nguồn đầu vào (webcam, video, ảnh, hoặc stream) bằng hàm determine_source_type
        self.source_type = self.determine_source_type()
        # Khởi tạo biến cap để đọc video hoặc webcam, ban đầu là None
        self.cap = None
        # Khởi tạo biến vid_writer để ghi video đầu ra, ban đầu là None
        self.vid_writer = None

    def determine_source_type(self):
        # Kiểm tra nếu nguồn là số (ví dụ '0'), thì đó là webcam
        if self.opt.source.isnumeric():
            return 'webcam'
        # Kiểm tra nếu nguồn là URL (bắt đầu bằng http:// hoặc https://), thì đó là stream
        if self.opt.source.lower().startswith(('http://', 'https://')):
            return 'stream'
        # Kiểm tra nếu nguồn là file tồn tại trên máy
        if Path(self.opt.source).exists():
            # Nếu file có đuôi .jpg, .png, .jpeg thì là ảnh
            if Path(self.opt.source).suffix.lower() in ['.jpg', '.png', '.jpeg']:
                return 'image'
            # Nếu không phải ảnh thì là video
            return 'video'
        # Nếu không thuộc trường hợp nào trên, báo lỗi
        raise ValueError("Nguồn đầu vào không hợp lệ")

    def crop_image(self, image, box):
        """Cắt ảnh trong bounding box của Violence"""
        # Lấy tọa độ bounding box từ box.xyxy[0], là tensor [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
        # Chuyển các tọa độ từ float sang int để cắt ảnh
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        # Cắt ảnh gốc (image) theo tọa độ bounding box, trả về vùng cắt và offset (xmin, ymin)
        return image[ymin:ymax, xmin:xmax], (xmin, ymin)

    def draw_skeleton(self, image, kpts, offset):
        """Vẽ khung xương lên ảnh gốc với offset của bounding box"""
        # Kiểm tra nếu không có keypoints hoặc keypoints rỗng thì thoát hàm
        if kpts is None or len(kpts.data) == 0:
            return
        # Lấy tensor keypoints từ kpts.data, shape: (num_persons, 17, 3)
        # - num_persons: số người được phát hiện
        # - 17: số keypoints (mũi, vai, tay, v.v.)
        # - 3: [x, y, confidence] cho mỗi keypoint
        keypoints_data = kpts.data
        # Lặp qua từng người trong danh sách keypoints
        for person_kpts in keypoints_data:
            # Lặp qua từng keypoint của người đó
            for kpt in person_kpts:
                # Nếu confidence > 0 (keypoint được phát hiện)
                if kpt[2] > 0:
                    # Tính tọa độ x, y trên ảnh gốc bằng cách cộng offset
                    x, y = int(kpt[0] + offset[0]), int(kpt[1] + offset[1])
                    # Vẽ một chấm tròn màu xanh tại tọa độ (x, y) trên ảnh gốc
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    def is_falling(self, person_kpts, person_box, frame_height):
        """Kiểm tra xem một người có đang té ngã hay không"""
        # person_kpts: tensor shape (17, 3) chứa [x, y, confidence] của 1 người
        # Lấy tọa độ y của vai trái (index 5) nếu confidence > 0
        left_shoulder_y = person_kpts[5][1] if person_kpts[5][2] > 0 else None
        # Lấy tọa độ y của vai phải (index 6) nếu confidence > 0
        right_shoulder_y = person_kpts[6][1] if person_kpts[6][2] > 0 else None

        # Tính trung bình tọa độ y của vai nếu phát hiện được
        if left_shoulder_y and right_shoulder_y:
            shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        elif left_shoulder_y:
            shoulder_y = left_shoulder_y
        elif right_shoulder_y:
            shoulder_y = right_shoulder_y
        else:
            # Nếu không phát hiện vai nào, trả về False (không té ngã)
            return False

        # person_box: tensor [xmin, ymin, xmax, ymax] của bounding box người
        xmin, ymin, xmax, ymax = person_box
        # Tính chiều rộng (dx) và chiều cao (dy) của bounding box
        dx = xmax - xmin
        dy = ymax - ymin
        # Tính difference = dy - dx để kiểm tra hình dạng bounding box
        difference = dy - dx
        # Đặt ngưỡng dựa trên chiều cao frame, ví dụ nửa frame + 100
        thre = (frame_height // 2) + 100

        # Kiểm tra điều kiện té ngã:
        # - difference <= 0 (chiều cao <= chiều rộng) và vai thấp (shoulder_y > thre)
        # - hoặc difference < 0 (chiều cao nhỏ hơn chiều rộng)
        if (difference <= 0 and shoulder_y > thre) or (difference < 0):
            return True
        return False

    def initialize_video_writer(self, frame):
        """Khởi tạo VideoWriter để lưu video"""
        # Chỉ khởi tạo nếu yêu cầu lưu video và chưa có vid_writer
        if self.opt.save_img and self.vid_writer is None:
            # Đặt FPS: 30 cho webcam, hoặc lấy từ video gốc
            fps = 30 if self.source_type == 'webcam' else int(self.cap.get(cv2.CAP_PROP_FPS))
            # Lấy chiều rộng và cao của frame
            w, h = frame.shape[1], frame.shape[0]
            # Tạo đường dẫn file video đầu ra với timestamp
            output_path = str(self.save_dir / f'output_{int(time.time())}.mp4')
            # Khởi tạo VideoWriter với codec mp4v, FPS và kích thước frame
            self.vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            print(f"Đã khởi tạo video writer tại: {output_path}")

    def process_image(self):
        # Đọc ảnh từ nguồn đầu vào (opt.source)
        image = cv2.imread(self.opt.source)
        # Chạy mô hình phát hiện bạo lực trên ảnh, trả về kết quả dự đoán
        violence_results = self.violence_model.predict(image, conf=self.opt.conf, imgsz=self.opt.imgsz)
        # Lấy ID của lớp 'violence' từ danh sách tên lớp của mô hình
        violence_class_id = next(k for k, v in self.violence_model.names.items() if v == 'violence')
        # Lọc các bounding box có nhãn 'violence'
        violence_boxes = [box for box in violence_results[0].boxes if box.cls == violence_class_id]

        # Xử lý từng bounding box bạo lực
        for box in violence_boxes:
            # Cắt ảnh theo bounding box, trả về ảnh cắt và offset
            cropped_image, offset = self.crop_image(image, box)
            # Chạy mô hình pose trên ảnh cắt để lấy keypoints và bounding box người
            pose_results = self.pose_model.predict(cropped_image, conf=self.opt.conf, imgsz=self.opt.imgsz)
            # Lấy tọa độ bounding box để vẽ lên ảnh gốc
            p1 = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
            p2 = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
            # Vẽ hình chữ nhật màu xanh cho Violence
            cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
            # Viết nhãn "Violence" phía trên bounding box
            cv2.putText(image, "Violence", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Nếu có kết quả pose và keypoints tồn tại
            if pose_results and pose_results[0].keypoints is not None:
                # Vẽ khung xương lên ảnh gốc
                self.draw_skeleton(image, pose_results[0].keypoints, offset)

        # Nếu yêu cầu hiển thị ảnh
        if self.opt.view_img:
            cv2.imshow("Violence Detection", image)
            cv2.waitKey(0)
        # Nếu yêu cầu lưu ảnh
        if self.opt.save_img:
            output_path = str(self.save_dir / Path(self.opt.source).name)
            cv2.imwrite(output_path, image)
            print(f"Ảnh đã lưu tại: {output_path}")

    def process_video_stream(self):
        # Mở nguồn video hoặc webcam (0 nếu là webcam, hoặc đường dẫn video)
        self.cap = cv2.VideoCapture(0 if self.source_type == 'webcam' else self.opt.source)
        # Kiểm tra nếu không mở được nguồn
        if not self.cap.isOpened():
            raise ValueError("Không thể mở nguồn video/webcam")

        # Vòng lặp xử lý từng frame
        while self.cap.isOpened():
            # Đọc frame từ video/webcam, success là True nếu đọc thành công
            success, frame = self.cap.read()
            if not success:
                print("Không thể đọc khung hình, thoát...")
                break

            # Lấy chiều cao frame để dùng trong kiểm tra té ngã
            frame_height = frame.shape[0]

            # Chạy mô hình phát hiện bạo lực trên frame
            violence_results = self.violence_model.predict(frame, conf=self.opt.conf, imgsz=self.opt.imgsz)
            # Lấy ID của lớp 'Violence' từ danh sách tên lớp
            violence_class_id = next(k for k, v in self.violence_model.names.items() if v == 'Violence')
            # Lọc các bounding box có nhãn 'Violence'
            violence_boxes = [box for box in violence_results[0].boxes if box.cls == violence_class_id]

            # Xử lý từng bounding box bạo lực
            for box in violence_boxes:
                # Cắt frame theo bounding box
                cropped_frame, offset = self.crop_image(frame, box)
                # Chạy mô hình pose trên frame cắt
                pose_results = self.pose_model.predict(cropped_frame, conf=self.opt.conf, imgsz=self.opt.imgsz)

                # Lấy tọa độ bounding box Violence để vẽ
                p1 = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
                p2 = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))

                # Khởi tạo nhãn và màu mặc định cho Violence
                label = "Violence"
                color = (0, 255, 0)  # Màu xanh

                # Kiểm tra té ngã nếu có keypoints và bounding box người
                if pose_results and pose_results[0].keypoints is not None and pose_results[0].boxes is not None:
                    # Lấy tensor keypoints, shape: (num_persons, 17, 3)
                    # - num_persons: số người trong frame cắt
                    # - 17: số keypoints
                    # - 3: [x, y, confidence]
                    kpts = pose_results[0].keypoints.data
                    # Lấy tensor bounding box người, shape: (num_persons, 4)
                    # - 4: [xmin, ymin, xmax, ymax]
                    boxes = pose_results[0].boxes.xyxy
                    # Kiểm tra nếu có bounding box người
                    if boxes.shape[0] > 0:
                        # Lặp qua từng người
                        for i in range(kpts.shape[0]):
                            if i < boxes.shape[0]:  # Đảm bảo không vượt quá số box
                                # Lấy keypoints của người thứ i
                                person_kpts = kpts[i]
                                # Sao chép bounding box và điều chỉnh với offset
                                person_box = boxes[i].clone()
                                person_box[0] += offset[0]  # xmin
                                person_box[1] += offset[1]  # ymin
                                person_box[2] += offset[0]  # xmax
                                person_box[3] += offset[1]  # ymax
                                # Kiểm tra nếu người này té ngã
                                if self.is_falling(person_kpts, person_box, frame_height):
                                    # Lấy tọa độ bounding box người để vẽ
                                    xmin, ymin, xmax, ymax = person_box.int().tolist()
                                    # Vẽ nhãn "Falling" màu đỏ
                                    cv2.putText(frame, "Falling", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                                (0, 0, 255), 2)
                                    # Vẽ bounding box đỏ cho người té ngã
                                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                                    # Cập nhật nhãn và màu cho Violence thành "Serious Violence"
                                    label = "Serious Violence"
                                    color = (0, 0, 255)

                # Vẽ bounding box và nhãn cho Violence
                cv2.rectangle(frame, p1, p2, color, 2)
                cv2.putText(frame, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Vẽ khung xương nếu có keypoints
                if pose_results and pose_results[0].keypoints is not None:
                    self.draw_skeleton(frame, pose_results[0].keypoints, offset)

            # Hiển thị frame nếu yêu cầu
            if self.opt.view_img:
                cv2.imshow("Real-time Violence Detection", frame)
            # Lưu frame vào video nếu yêu cầu
            if self.opt.save_img:
                self.initialize_video_writer(frame)
                if self.vid_writer is not None:
                    self.vid_writer.write(frame)
                else:
                    print("Lỗi: VideoWriter chưa được khởi tạo!")

            # Thoát nếu nhấn phím 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Giải phóng tài nguyên khi kết thúc
        if self.vid_writer:
            self.vid_writer.release()
            print("Đã lưu video và giải phóng VideoWriter")
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        # Chạy xử lý dựa trên loại nguồn đầu vào
        if self.source_type == 'image':
            self.process_image()
        else:
            self.process_video_stream()


def parse_args():
    import argparse
    # Tạo parser để phân tích tham số dòng lệnh
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
    # Trả về các tham số đã phân tích
    return parser.parse_args()


if __name__ == "__main__":
    # Phân tích tham số dòng lệnh
    opt = parse_args()
    # Khởi tạo hệ thống với tham số
    detector = ViolencePoseDetectionSystem(opt)
    # Chạy hệ thống
    detector.run()