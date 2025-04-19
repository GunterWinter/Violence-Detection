import cv2

# Mở video gốc
cap = cv2.VideoCapture('data/data2/Violence/cam2/13.mp4')

# Lấy FPS từ video gốc
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 18  # Đặt mặc định là 18 nếu không lấy được (dựa trên giả định của bạn)
fps = int(fps)
print(f"FPS của video: {fps}")

# Lấy kích thước khung hình
ret, frame = cap.read()
if not ret:
    print("Không thể đọc video")
    exit()
h, w = frame.shape[:2]
print(f"Kích thước khung hình: {w}x{h}")

# Tạo video writer với FPS và kích thước đúng
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# Quay lại đầu video và ghi
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    frame_count += 1

print(f"Đã ghi {frame_count} khung hình")

# Giải phóng tài nguyên
cap.release()
out.release()