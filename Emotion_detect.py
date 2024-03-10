import cv2
import numpy as np

def detect_emotion(face, emotion_net, emotion_labels):
    # Kiểm tra xem ảnh face có trống không
    if face.size == 0:
        print("Ảnh face trống")
        return None

    # Thay đổi kích thước và chuyển đổi màu ảnh
    resized_face_emotion = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
    resized_face_emotion = cv2.cvtColor(resized_face_emotion, cv2.COLOR_BGR2GRAY)
    resized_face_emotion = np.expand_dims(resized_face_emotion, axis=-1)
    resized_face_emotion = resized_face_emotion.astype('float32') / 255.0
    resized_face_emotion = np.expand_dims(resized_face_emotion, axis=0)

    # Thực hiện dự đoán cảm xúc
    emotion_preds = emotion_net.predict(resized_face_emotion)

    # Đảm bảo emotion_preds không trống trước khi sử dụng np.argmax
    if emotion_preds.any():
        emotion_label = emotion_labels[np.argmax(emotion_preds)]
        return emotion_label
    else:
        return "Unknown"
