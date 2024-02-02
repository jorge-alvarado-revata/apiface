from fastapi import FastAPI
from pydantic import BaseModel
from typing import Set, Union

app = FastAPI()

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