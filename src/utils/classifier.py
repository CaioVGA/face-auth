# from random import randint

import cv2

from utils.recognizer import FaceRecognizer
from utils.editor import ImageEditor

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainning.yml")
editor = ImageEditor()

class CascadeClassifier():

    def __init__(self) -> None:
        self.faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
        self.eyeCascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
        self.mouthCascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

    def faceDetection(self, image, auth_variable=False):

        recognizer = FaceRecognizer()

        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(image, scaleFactor=1.5, minNeighbors=5)

        for (x, y, w, h) in faces:

            roi_gray = gray_frame[y:y+h, x:x+w]  # (ycord_start, ycord_end)
            roi_color = image[y:y+h, x:x+w]  # (ycord_start, ycord_end)
            id_, conf = recognizer.prediction(roi_gray)
            
            labels = recognizer.labelPickle()

            if conf >= 60 and conf <= 90:
                print(f"Recognized: {labels[id_]}, conf: {conf}")
                editor.putTextLabel(frame=image, name=labels[id_], cord=[x,y])
                auth_variable = True
            print(auth_variable)
            # img_item = f"src/images/{labels[id_]}/img{randint(0,100)}.png"
            img_item = f"src/images/last-face/img.png"
            cv2.imwrite(img_item, roi_color)
            editor.rectangleDraw(frame=image, cord=[x, y, w, h])

            # Detecting eyes and mouth from any face
            self.subItemsDetection(self.eyeCascade, roi_color, roi_gray, region_draw=True)
            self.subItemsDetection(self.mouthCascade, roi_color, roi_gray, region_draw=True)

        return image, auth_variable    

    def subItemsDetection(self, classifier, bgr_image, gray_image, region_draw=False): 

        subitem = classifier.detectMultiScale(gray_image, scaleFactor=1.5, minNeighbors=5)
        
        if region_draw is True:
            for (x, y, w, h) in subitem:
                editor.rectangleDraw(bgr_image, [x,y,w,h], color=(255,0,0))