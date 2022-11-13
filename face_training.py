import pickle
import cv2
import os
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath((__file__)))
image_dir = os.path.join(BASE_DIR, 'images')

face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()



current_id = 0
labels_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(' ', '-') #.lower# наш человек (папка)

            print(label, path)
            if label in labels_ids:
                pass
            else:
                labels_ids[label] = current_id
                current_id += 1
            id_ = labels_ids[label]
            print(labels_ids)

            y_labels.append(label) # какое то число
            x_train.append(path)# в нампай массив
            pil_image = Image.open(path).convert('L')   # серый слой
            image_array = np.array(pil_image, "uint8")
            print(image_array)
            faces = face_cascade_db.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

"""
for image in os.listdir(BASE_DIR):
  im = Image.open(BASE_DIR)
  # If is png image
  if im.format is 'PNG':
    # and is not RGBA
    if im.mode is not 'RGBA':
      im.convert("RGBA").save(f"{BASE_DIR}2.png")
"""

#print(y_labels)
#print(x_train)

with open('labels.pickle', 'wb') as f:
    pickle.dump(labels_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save('trainner.yml')
