import cv2
import face_recognition
import os
from datetime import datetime

def capture_images(person_name, num_images=10):
    # Tạo một thư mục cho ảnh của người đó
    person_dir = os.path.join("database", person_name)
    os.makedirs(person_dir, exist_ok=True)

    # Mở webcam
    cap = cv2.VideoCapture(0)

    print(f"Đang chụp {num_images} ảnh cho {person_name}...")

    count = 0
    while count < num_images:
        # Chụp frame từ webcam
        ret, frame = cap.read()

        # Tìm vị trí khuôn mặt
        face_locations = face_recognition.face_locations(frame)

        if face_locations:
            # Vẽ hình chữ nhật quanh khuôn mặt
            top, right, bottom, left = face_locations[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Lưu ảnh
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            image_filename = f"face_detect_{count + 1}.jpg"
            image_path = os.path.join(person_dir, image_filename)
            cv2.imwrite(image_path, frame)

            print(f"Đã chụp ảnh {count + 1}!")

            count += 1

        # Hiển thị frame kết quả
        cv2.imshow("Capture", frame)

        # Thoát khỏi vòng lặp khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng webcam và đóng tất cả cửa sổ
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Nhập tên của người cần nhận diện khuôn mặt
    person_name = input("Nhập tên của người cần nhận diện khuôn mặt: ")

    # Chụp 10 ảnh cho người đó
    capture_images(person_name, num_images=50)

    print(f"Quá trình nhận diện khuôn mặt cho {person_name} đã hoàn thành.")
