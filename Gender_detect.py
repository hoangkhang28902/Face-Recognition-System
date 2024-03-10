
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

def detect_gender(face, gender_net, gender_list, model_mean_values):
    # Kiểm tra xem ảnh face có trống không
    if face.size == 0:
        print("Ảnh face trống")
        return None

    # Tạo blob từ ảnh face
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), model_mean_values, swapRB=False)

    # Đặt đầu vào cho mô hình giới tính và thực hiện dự đoán
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()

    # Đảm bảo gender_preds không trống trước khi sử dụng np.argmax
    if gender_preds.any():
        gender = gender_list[gender_preds[0].argmax()]
        return gender
    else:
        return "Unknown"
