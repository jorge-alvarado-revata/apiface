from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Set, Union, Annotated
import numpy as np

import cv2



app = FastAPI()

origins = [
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



@app.post("/comparison/")
async def post_compasison(item: ComparaImage):
    item.result = 60.6
    results = {"status_code":200, "status_result": "OK.", "result": item }
    return results

@app.get("/detection/{url_foto}")
async def face_compasison():
    results = {"status_code":200, "status_result": "OK." }
    return results

@app.post("/comparison_img/")
async def create_file(
    fileA: Annotated[UploadFile, File()],
    fileB: Annotated[UploadFile, File()]):
    
    imageA =  fileA.file.read()

    imageB =  fileB.file.read()

    imgA_cv2 = cv2.imdecode(np.frombuffer(imageA, np.uint8), cv2.IMREAD_UNCHANGED)

    imgB_cv2 = cv2.imdecode(np.frombuffer(imageB, np.uint8), cv2.IMREAD_UNCHANGED)



    return {
        "file_sizeA": imgA_cv2.shape,
        "file_sizeB": imgB_cv2.shape,
        "eval": 0.0
    }