import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
import numpy as np
import cv2
from urllib.parse import urlparse
from io import BytesIO
from os import listdir
from os.path import isfile, join

# To install this module, run:
# python -m pip install Pillow
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person


#  my face api key
KEY = "b9a212691cbf444e8d70c157cc2c454e"

# This endpoint of my face api
ENDPOINT = "https://sdcreations.cognitiveservices.azure.com/"


# Create an authenticated FaceClient.
face_client = FaceClient(
    "https://sdcreations.cognitiveservices.azure.com/",
    CognitiveServicesCredentials("b9a212691cbf444e8d70c157cc2c454e"),
)

face_classifier = cv2.CascadeClassifier(
    "Haarcascades/haarcascade_frontalface_default.xml"
)

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        cropped_face = img[y : y + h, x : x + w]

    return cropped_face


# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = "./faces/user/" + str(count) + ".jpg"
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(
            face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow("Face Cropper", face)

    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")


# ***************************************train  model ***************************************

data_path = "./faces/user/"
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer
model = cv2.face.LBPHFaceRecognizer_create()

# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()

# Let's train our model
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained successfully")


# ***********************run facial recognition ***************************


def face_detector(img, size=0.5):

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y : y + h, x : x + w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


# Open Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = model.predict(face)

        if results[1] < 500:
            confidence = int(100 * (1 - (results[1]) / 400))
            display_string = str(confidence) + "% Confident it is User"

        cv2.putText(
            image,
            display_string,
            (100, 120),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 120, 150),
            2,
        )

        if confidence > 75:
            cv2.putText(
                image,
                "you can proceed to transaction",
                (50,450),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Face Recognition", image)
        else:
            cv2.putText(
                image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2
            )
            cv2.imshow("Face Recognition", image)

    except:
        cv2.putText(
            image,
            "No Face Found",
            (220, 120),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2
        )
        cv2.imshow("Face Recognition", image)
        pass

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
