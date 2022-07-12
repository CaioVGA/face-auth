import cv2
import pickle

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