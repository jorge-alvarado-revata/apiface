from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Set, Union, Annotated
import face_recognition
import logging
from fastapi.encoders import jsonable_encoder

import numpy as np

import base64

import cv2

app = FastAPI()

origins = [
    "http://0.0.0.0",
    "http://0.0.0.0:80",
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Image(BaseModel):
    name: str
    url: str


class ComparaImage(BaseModel):
    item1: Image
    item2: Image
    result: float



def find_face_encoding(image):
    face_enc = face_recognition.face_encodings(image)
    return face_enc[0]



@app.post("/comparison_img/")
async def create_file(
    fileA: Annotated[UploadFile, File()],
    fileB: Annotated[UploadFile, File()]):
    
    imageA =  fileA.file.read()

    imageB =  fileB.file.read()

    imgA_cv2 = cv2.imdecode(np.frombuffer(imageA, np.uint8), cv2.IMREAD_UNCHANGED)

    imgB_cv2 = cv2.imdecode(np.frombuffer(imageB, np.uint8), cv2.IMREAD_UNCHANGED)

    gray_imgA = cv2.cvtColor(imgA_cv2, cv2.COLOR_BGR2RGB)

    gray_imgB = cv2.cvtColor(imgB_cv2, cv2.COLOR_BGR2RGB)

    image_1  = find_face_encoding(gray_imgA)

    image_2 = find_face_encoding(gray_imgB)

    #check si son las mismas personas

    is_same = face_recognition.compare_faces([image_1], image_2)[0]

    accuracy = 0.0

    if is_same:
        distance = face_recognition.face_distance([image_1], image_2)
        distance = round(distance[0] * 100)
        accuracy = 100 - round(distance)
    else:
        is_same = False

    return {
        "is_same": f"{is_same}",
        "accuracy": f"{accuracy}"
    }



@app.post("/detection_face/")
async def create_file(
    fileA: Annotated[UploadFile, File()]):
    
    imageA =  fileA.file.read()

    imgA_cv2 = cv2.imdecode(np.frombuffer(imageA, np.uint8), cv2.IMREAD_UNCHANGED)

    gray_imgA = cv2.cvtColor(imgA_cv2, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    

    face = face_classifier.detectMultiScale(
        gray_imgA, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    eyes = eye_cascade.detectMultiScale(gray_imgA, 1.3, 5)

    has_human = len(face)



    for (x, y, w, h) in face:
        cv2.rectangle(gray_imgA, (x, y), (x + w, y + h), (0, 255, 0), 4)

    for(ex, ey, ew, eh) in eyes:
        cv2.rectangle(gray_imgA,(ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)


    img_rgb = cv2.cvtColor(gray_imgA, cv2.COLOR_BGR2RGB)

    _, encoded_img = cv2.imencode('.png', img_rgb)

    encoded_img = base64.b64encode(encoded_img)

    return {
        "content": encoded_img,
        "has_person": has_human
    }