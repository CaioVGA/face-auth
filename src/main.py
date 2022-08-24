import cv2
from time import time
from imutils.video import VideoStream, FPS
from utils.classifier import CascadeClassifier
from raspberry.gpioControl import RaspPinout

# instancia do classificador e da Raspberry
classifier = CascadeClassifier()
rasp = RaspPinout()
# instancia da captura de video
stream = VideoStream(src=0).start()
time.sleep(1)
fps = FPS.start()

# inicializacao do loop infinito
while True:

    # leitura das imagens fornecidas pela camera
    frame = stream.read()

    # aplicacao da imagem da camera para o modelo
    faces, auth = classifier.faceDetection(image=frame)
    rasp.control(authentication=auth)
    # mostra a saída com o reconhecimento facial
    cv2.imshow('Face Recognizer', faces)

    # condicional para quebra do loop infinito
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# liberacao da camera e fecha as janelas
stream.release()
cv2.destroyAllWindows()
