import cv2
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainning.yml")

class CascadeClassifier():

    def __init__(self) -> None:
        self.cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

    def rectangleDraw(self, frame, cord: list):

        color = (0, 255, 0)  # BGR
        stroke = 2
        end_cord_x = cord[0] + cord[2]  # x + w
        end_cord_y = cord[1] + cord[3]  # y + h
        rectangle = cv2.rectangle(frame,
                                  (cord[0], cord[1]),
                                  (end_cord_x, end_cord_y),
                                  color,
                                  stroke)
        return rectangle

    def faceDetection(self, image, auth_variable=False):
        recognizer = FaceRecognizer()

        faces = self.cascade.detectMultiScale(image, scaleFactor=1.5, minNeighbors=5)

        for (x, y, w, h) in faces:

            roi = image[y:y+h, x:x+w]  # (ycord_start, ycord_end)
            id_, conf = recognizer.prediction(roi)
            
            labels = recognizer.labelPickle()

            if conf >= 45 and conf <= 85:
                print(f"Recognized: {labels[id_]}")
                auth_variable = True
            print(auth_variable)
            img_item = f"images/last-face/face.png"
            cv2.imwrite(img_item, roi)
            self.rectangleDraw(frame=image, cord=[x, y, w, h])

        return image, auth_variable

class FaceRecognizer():
    
    def __init__(self) -> None:
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    def prediction(self, roi):
        self.recognizer.read("trainning.yml")

        id_, conf  = self.recognizer.predict(roi)
        return id_, conf

    def labelPickle(self):
        labels = {"person_name": 1}

        with open("labels.pickle", 'rb') as file:
            og_labels = pickle.load(file)
            labels = {v:k for k,v in og_labels.items()}
        return labels