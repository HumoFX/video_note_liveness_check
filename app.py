from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from face_anti_spoofing import check_liveness

app = FastAPI()


class Video(BaseModel):
    name: str
    question: int


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/liveness")
def check_liveness_view(video: Video):
    print(video)
    passed, get_photo, photo_name = check_liveness(video.name, video.question)
    return {"check": passed, "get_photo": get_photo, "photo_name": photo_name}
    # return {"name": video.name, "question": video.question}
