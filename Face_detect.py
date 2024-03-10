import os
import glob
import cv2
import numpy as np
import face_recognition

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25

    def load_encoding_images(self, dataset_path):
        """
        Load encoding images from path
        :param dataset_path: Đường dẫn tương đối đến thư mục chứa dataset
        :return: None
        """
        # Tạo đường dẫn tương đối đến thư mục dataset
        dataset_path = os.path.join(os.path.dirname(__file__), dataset_path)

        # Sử dụng glob để lấy tất cả đường dẫn ảnh trong thư mục dataset và các thư mục con của nó
        images_path = glob.glob(os.path.join(dataset_path, "*", "*.jpg"))

        print("Tổng cộng {} ảnh được tìm thấy trong database.".format(len(images_path)))

        # Lưu trữ encoding và tên của ảnh
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Lấy tên file từ đường dẫn ban đầu
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            # Lấy tên của người từ tên thư mục chứa ảnh
            person_name = os.path.basename(os.path.dirname(img_path))

            try:
                # Lấy encoding với ít landmarks và jitters hơn
                img_encoding = face_recognition.face_encodings(rgb_img, model="small", num_jitters=1)[0]

                # Lưu trữ tên file và encoding của file
                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(person_name)
            except IndexError:
                pass

        print("Encoding của ảnh đã được tải")

    def recognize_face(self, face):
        # Thực hiện xử lý nhận diện khuôn mặt ở đây
        # Ví dụ: trả về tên người dựa trên quy tắc nhất định hoặc một quy trình nhận diện cụ thể
        # Thay thế dòng dưới bằng mã của bạn để nhận diện khuôn mặt và trả về tên
        recognized_name = "Unknown"

        return recognized_name

    # def detect_known_faces(self, frame):
    #     small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
    #     rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    #     face_locations = face_recognition.face_locations(rgb_small_frame)
    #     face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    #     face_names = []
    #     for face_encoding in face_encodings:
    #         matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
    #         name = "Unknown"
    #         face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
    #         best_match_index = np.argmin(face_distances)
    #         if matches[best_match_index]:
    #             name = self.known_face_names[best_match_index]
    #         face_names.append(name)

    #     # Convert vị trí khuôn mặt từ small frame sang frame gốc
    #     face_locations = np.array(face_locations)
    #     face_locations = face_locations / self.frame_resizing

    #     return face_locations.astype(int), face_names

    # def detect_known_faces(self, frame):
    #         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         face_locations = face_recognition.face_locations(rgb_frame)
    #         face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    #         face_names = []
    #         for face_encoding in face_encodings:
    #             matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
    #             name = "Unknown"
    #             face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
    #             best_match_index = np.argmin(face_distances)
    #             if matches[best_match_index]:
    #                 name = self.known_face_names[best_match_index]
    #             face_names.append(name)

    #         # Convert vị trí khuôn mặt từ small frame sang frame gốc
    #         face_locations = np.array(face_locations)

    #         return face_locations.astype(int), face_names
    def detect_known_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            # Kiểm tra xem có ít nhất một khuôn mặt đã được nhận diện hay không
            if True in matches:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
            
            face_names.append(name)

        # Convert vị trí khuôn mặt từ small frame sang frame gốc
        face_locations = np.array(face_locations)

        return face_locations.astype(int), face_names