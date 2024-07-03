import cv2
import numpy as np

# Mở video từ đường dẫn đã chỉ định
video = cv2.VideoCapture(r'F:\IUH\NAM3_HK1\XU_LY_ANH\CUOI_KY\visiontraffic.avi')
# Lấy số khung hình trên giây (fps) của video
fps = int(video.get(cv2.CAP_PROP_FPS))
print(fps)

# Thiết lập diện tích nhỏ nhất của contour để được xem là một xe
minArea = 100
# Đếm số lượng xe trong toàn bộ video
count_car_in_video = 0
# Biến tạm để đếm số lượng xe trong khung hình trước
count_temp = 0
# Đếm số lượng xe trong khung hình hiện tại
count_car_in_frame = 0

# Khung hình trước đó để tính toán sự khác biệt
prev_frame = None

# Vòng lặp xử lý từng khung hình của video
while (True):
    # Đọc khung hình từ video
    ret, frame = video.read()
    # Kiểm tra nếu không đọc được khung hình
    if not ret:
        print('Fail to read video')
        break
    # Thay đổi kích thước khung hình
    frame = cv2.resize(frame, (1280, 720))
    # Chuyển đổi khung hình sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Làm mờ ảnh xám để giảm nhiễu
    gray = cv2.medianBlur(gray, 9)
    # Nếu không có khung hình trước đó, lưu khung hình hiện tại và tiếp tục
    if prev_frame is None:
        prev_frame = gray
        continue
    # Tính toán sự khác biệt giữa khung hình hiện tại và khung hình trước đó
    delta = cv2.absdiff(gray, prev_frame)
    # Chuyển đổi sự khác biệt thành ảnh nhị phân
    _, thresh_image = cv2.threshold(delta, 15, 255, cv2.THRESH_BINARY)
    # Làm mòn ảnh nhị phân
    pre_process_image = cv2.erode(thresh_image, np.ones((5, 5), np.uint8), iterations=1)
    # Giãn nở ảnh nhị phân
    pre_process_image = cv2.dilate(pre_process_image, np.ones((7, 7), np.uint8), iterations=5)
    # Đóng ảnh nhị phân
    pre_process_image = cv2.morphologyEx(pre_process_image, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    # Tìm các đường viền trong ảnh nhị phân
    contours, _ = cv2.findContours(pre_process_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Vòng lặp qua các đường viền tìm được
    for contour in contours:
        # Lấy tọa độ và kích thước của contour
        (x, y, w, h) = cv2.boundingRect(contour)
        # Tính toán diện tích của contour
        contourArea = cv2.contourArea(contour)
        # Bỏ qua các contour có diện tích nhỏ hơn diện tích nhỏ nhất
        if contourArea < minArea:
            continue
        # Vẽ hình chữ nhật bao quanh contour
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.rectangle(pre_process_image, (x, y), (x + w, y + h), (255, 255, 255), 3)

    # Kiểm tra sự thay đổi số lượng contour để đếm xe
    if (count_temp != len(contours)) & (count_temp < len(contours)):
        count_car_in_video += 1

    # Cập nhật số lượng xe tạm thời
    count_temp = len(contours)
    # Cập nhật số lượng xe trong khung hình hiện tại
    count_car_in_frame = len(contours)
    # Hiển thị số lượng xe trên khung hình
    cv2.putText(frame, "So luong xe xuat hien trong khung hinh: {}".format(count_car_in_frame), (1, 700), cv2.FONT_ITALIC, 1, (0, 0, 255), 3)
    # Hiển thị kết quả
    cv2.imshow("Chuong trinh dem so luong xe trong khung hinh", frame)
    # Cập nhật khung hình hiện tại làm khung hình trước đó
    prev_frame = gray

    # Nếu nhấn phím ESC, thoát vòng lặp
    if cv2.waitKey(40) == 27:
        break

# Giải phóng tài nguyên video và đóng cửa sổ hiển thị
video.release()
cv2.destroyAllWindows()
