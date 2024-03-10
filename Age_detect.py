
import cv2
from keras.models import load_model
from keras_preprocessing.image import img_to_array
import numpy as np

def detect_age(face, age_net, age_list, model_mean_values):
    # Kiểm tra xem ảnh face có trống không
    if face.size == 0:
        print("Ảnh face trống")
        return None

    # Tạo blob từ ảnh face
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), model_mean_values, swapRB=False)

    # Đặt đầu vào cho mô hình tuổi và thực hiện dự đoán
    age_net.setInput(blob)
    age_preds = age_net.forward()

    # Lấy index có giá trị lớn nhất trong age_preds
    age_index = age_preds[0].argmax()

    # Đảm bảo age_index nằm trong khoảng của danh sách tuổi
    if 0 <= age_index < len(age_list):
        age = age_list[age_index]
        return age
    else:
        return "Unknown"
