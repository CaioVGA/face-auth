import os
from PIL import Image
import pickle

import cv2
import numpy as np


# mapeamento dos diretórios com amostras de imagens
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

# importação dos kernels haar para a classificação
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id_number = 0
label_ids = dict() # id de cada usuário
x_train = list() # dados reservados para o treinamento do modelo
y_labels = list() # rótulo de identificação

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            if not label in label_ids:
                label_ids[label] = current_id_number
                current_id_number += 1
            id_ = label_ids[label]
            pil_image = Image.open(path).convert("L") 
            size = (550, 550)
            final_image = pil_image.resize(size, Image.Resampling.LANCZOS)
            image_array = np.array(final_image, "uint8")
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            # delimita a região de interesse de identificação
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open("labels.pickle", 'wb') as file:
    pickle.dump(label_ids, file)

# treinamento do modelo a partir dos dados coletados e representação do mesmo
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainning.yml") # output: trainning.yml