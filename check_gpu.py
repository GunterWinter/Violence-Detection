import torch
print(torch.cuda.is_available())
print(torch.version.cuda)

import onnxruntime as ort

model_path = r'C:\BaiTap\Python\Violence_Detection\Yolo11_Violence_Detection\runs\detect\train\weights\best.pt'

# Tạo session options
session_options = ort.SessionOptions()

try:
    session = ort.InferenceSession(model_path, session_options, providers=['CUDAExecutionProvider'])
    print("Mô hình đang chạy trên GPU với CUDAExecutionProvider.")
except Exception as e:
    print("Không thể sử dụng GPU. Lỗi:", e)
    session = ort.InferenceSession(model_path, session_options, providers=['CPUExecutionProvider'])
    print("Mô hình đang chạy trên CPU.")

available_providers = session.get_providers()
print("Các provider hiện có:", available_providers)

if 'CUDAExecutionProvider' in available_providers:
    print("Xác nhận: Mô hình đang chạy trên GPU.")
else:
    print("Xác nhận: Mô hình đang chạy trên CPU.")