import os
from PIL import Image
import pickle

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id_number = 0
label_ids = dict()
x_train = list()
y_labels = list()

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            if not label in label_ids:
                label_ids[label] = current_id_number
                current_id_number += 1
            id_ = label_ids[label]
            # verifica a imagem em questão, transforma em uma lista do tipo numpy (gray scale).
            # x_train.append(path) 
            # y_labels.append(label)
            pil_image = Image.open(path).convert("L") # grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.Resampling.LANCZOS)
            image_array = np.array(final_image, "uint8")
            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# print(y_labels)
# print(x_train)

with open("labels.pickle", 'wb') as file:
    pickle.dump(label_ids, file)

# Training the model with machine learning method
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainning.yml") # save as a file