# Face-trainer
import os
import numpy as np
from PIL import Image
import cv2
import pickle
face_cascade = cv2.CascadeClassifier(
    'C:\\Users\\ramud\\Desktop\\PYTHON\\BIG_PROJECT\\haarcascade_frontalface_default.xml')
img_dir = os.path.join(os.getcwd(), "faces")
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}
y_labels = []
x_train = []


for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            # print(label,path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]
            # print(label_ids)
            # y_labels.append(label)
            # x_train.append(path) # we need to convert the image into a numpy array , for the face comparison
            # convert() will conver the image into grayscale
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")
            # print(image_array)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# print(y_labels)
# print(x_train)

with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
