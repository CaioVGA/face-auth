from this import d
from tkinter import W
from turtle import width
import cv2


class CascadeClassifier():

    def __init__(self) -> None:
        self.cascade = cv2.CascadeClassifier(
            'cascades/data/haarcascade_frontalface_alt2.xml')

    def colorConvertion(self, image, gray: bool):
        """
        It converts a color to another scale

        Args:
            color_type (str): Specifies the scale
        """

        if gray is True:

            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        else:
            img = image

        return img

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

    def faceDetection(self, image):

        faces = self.cascade.detectMultiScale(
            image, scaleFactor=1.5, minNeighbors=5)

        for (x, y, w, h) in faces:

            roi = image[y:y+h, x:x+w]  # (ycord_start, ycord_end)
            img_item = f"images/face.png"
            cv2.imwrite(img_item, roi)
            self.rectangleDraw(frame=image, cord=[x, y, w, h])
