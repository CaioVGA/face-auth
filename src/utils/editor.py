import cv2

class ImageEditor():

    def rectangleDraw(self, frame, cord: list, color = (0, 255, 0)):
        
        stroke = 2
        end_cord_x = cord[0] + cord[2]  # x + w
        end_cord_y = cord[1] + cord[3]  # y + h
        rectangle = cv2.rectangle(frame,
                                  (cord[0], cord[1]),
                                  (end_cord_x, end_cord_y),
                                  color,
                                  stroke)
        return rectangle
    
    def putTextLabel(self, frame, name, cord: list):

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        stroke = 2
        cv2.putText(frame, name, (cord[0], cord[1]), font, 1, color, stroke, cv2.LINE_AA)