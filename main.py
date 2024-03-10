
import argparse
import cv2
from Age_detect import detect_age
from Gender_detect import detect_gender
from Emotion_detect import detect_emotion
from Face_detect import SimpleFacerec
from keras.models import load_model
from keras_preprocessing.image import img_to_array
import numpy as np

def detect_and_display_results(frame, age_net, age_list, gender_net, gender_list, emotion_net, emotion_labels, model_mean_values, padding, sfr):
    face_boxes = []
    
    face_proto = "model/opencv_face_detector.pbtxt"
    face_model = "model/opencv_face_detector_uint8.pb"
    face_net = cv2.dnn.readNet(face_model, face_proto)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            face_boxes.append(box.astype("int"))

    for i, face_box in enumerate(face_boxes):
        face = frame[max(0, face_box[1] - padding): min(face_box[3] + padding, frame.shape[0] - 1),
                     max(0, face_box[0] - padding): min(face_box[2] + padding, frame.shape[1] - 1)]

        if face.size == 0:  # Kiểm tra xem face có kích thước hợp lý hay không
            continue
        
        gender = detect_gender(face, gender_net, gender_list, model_mean_values)
        print(f'Gender: {gender}')

        age = detect_age(face, age_net, age_list, model_mean_values)
        print(f'Age: {age} years')

        emotion = detect_emotion(face, emotion_net, emotion_labels)
        print(f'Emotion: {emotion}')

        face_names = sfr.detect_known_faces(face)
        print(f'Tên: {face_names}')

        cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 1
        
        info_text_gender_age = f'Gender: {gender}, Age: {age}'
        cv2.putText(frame, info_text_gender_age, (face_box[0], face_box[1] - 30),
                    font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

        info_text_emotion = f'Emotion: {emotion}'
        cv2.putText(frame, info_text_emotion, (face_box[0], face_box[1] - 10),
                    font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

        for name in face_names[1]:
            folder_name = str(name)  # Chuyển đổi name thành chuỗi
            name_text = f'Name: {folder_name}'
            print(name_text)
            cv2.putText(frame, name_text, (face_boxes[i][0], face_boxes[i][1] - 50),
                        font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

    return frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')

    args = parser.parse_args()

    age_proto = "model/age_deploy.prototxt"
    age_model = "model/age_net.caffemodel"
    gender_proto = "model/gender_deploy.prototxt"
    gender_model = "model/gender_net.caffemodel"

    model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    gender_list = ['Male', 'Female']

    age_net = cv2.dnn.readNet(age_model, age_proto)
    gender_net = cv2.dnn.readNet(gender_model, gender_proto)

    emotion_model = "model/emotion_detection.h5"
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotion_net = load_model(emotion_model)

    sfr = SimpleFacerec()
    sfr.load_encoding_images("database/")

    cv2.namedWindow('Multifunctional Detection', cv2.WINDOW_NORMAL)
    
    video = cv2.VideoCapture(args.image if args.image else 0)
    padding = 20

    while True:
        has_frame, frame = video.read()
        if not has_frame:
            cv2.waitKey()
            break
        
        result_img = detect_and_display_results(frame, age_net, age_list, gender_net, gender_list, emotion_net, emotion_labels, model_mean_values, padding, sfr)
        
        # Lấy kích thước thực của khung hình
        frame_height, frame_width, _ = frame.shape
        cv2.imshow("Multifunctional Detection", result_img)
        cv2.resizeWindow("Multifunctional Detection", frame_width, frame_height)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
