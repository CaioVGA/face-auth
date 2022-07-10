import cv2
from utils.classifier import CascadeClassifier


classifier = CascadeClassifier()
stream = cv2.VideoCapture(0)

while True:

    # Capturing frame-by-frame
    ret, frame = stream.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Classifier
    faces, auth = classifier.faceDetection(image=gray_frame)

    # Display the result of capture
    cv2.imshow('Face Recognizer', faces)

    # Waiting press a key to quit the program
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Release the capture
stream.release()
cv2.destroyAllWindows()
