from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
import random
import cv2
import imutils
import f_liveness_detection
import questions
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import os
import tensorflow as tf
import aiohttp
tf.get_logger().setLevel('INFO')


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def show_image(cam, text, color=(0, 0, 255)):
    ret, im = cam.read()
    # im = imutils.resize(im, width=348)
    # im = cv2.flip(im, 1)
    cv2.putText(im, text, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
    return im


async def check_liveness(video_name, question_id):
    # Load the detector
    EYE_AR_THRESH = 0.35
    detector = dlib.get_frontal_face_detector()

    # Load the predictor
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    predictor = dlib.shape_predictor("blink_detection/model_landmarks/shape_predictor_68_face_landmarks.dat")
    # cv2.namedWindow('liveness_detection')
    cam = cv2.VideoCapture(video_name)
    # cam = cv2.VideoCapture(0)
    video_name = video_name.split(".")[0]
    # video_name = video_name.split("\\")[-1]
    print("Video name: ", video_name)
    COUNTER, TOTAL = 0, 0
    counter_ok_questions = 0
    counter_ok_consecutives = 0
    limit_consecutives = 1
    limit_questions = 1
    counter_try = 0
    limit_try = 100
    result = False
    question = questions.question_bank(question_id)
    get_photo = False
    photo_name = ""
    blinks_up = 0
    blink = 0
    for i_try in range(limit_try):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # <----------------------- ingestar data
        ret, im = cam.read()
        if ret:
            # im = imutils.resize(im, width=1440)
            gray = cv2.cvtColor(src=im, code=cv2.COLOR_BGR2GRAY)
            face = detector(gray)
            landmarks = predictor(image=gray, box=face[0])
            shape = face_utils.shape_to_np(landmarks)
            right_ear_bot = landmarks.part(14)
            right_ear_up = landmarks.part(16)
            left_ear_bot = landmarks.part(2)
            left_ear_up = landmarks.part(0)
            nose_bot = landmarks.part(30)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0
            print("ear: ", ear)
            if not get_photo:

                nose_bot_right_ear_bot = (
                        ((right_ear_bot.x - nose_bot.x) ** 2 + (right_ear_bot.y - nose_bot.y) ** 2) ** 0.5)
                nose_bot_left_ear_bot = (
                        ((left_ear_bot.x - nose_bot.x) ** 2 + (left_ear_bot.y - nose_bot.y) ** 2) ** 0.5)
                nose_bot_right_ear_up = (
                        ((right_ear_up.x - nose_bot.x) ** 2 + (right_ear_up.y - nose_bot.y) ** 2) ** 0.5)
                nose_bot_left_ear_up = (((left_ear_up.x - nose_bot.x) ** 2 + (left_ear_up.y - nose_bot.y) ** 2) ** 0.5)
                # check if abs difference is less than 50% between distance left and right ear with distance between right and left ear
                if abs((nose_bot_right_ear_bot - nose_bot_left_ear_bot) / nose_bot_right_ear_bot) <= 0.5 and \
                        abs((nose_bot_right_ear_up - nose_bot_left_ear_up) / nose_bot_right_ear_up) <= 0.5 and \
                        ear >= EYE_AR_THRESH:
                    get_photo = True
                    photo_name = f"{video_name}_{COUNTER}.jpg"
                    # extract head
                    head = im[face[0].top() - 85:face[0].bottom() + 16, face[0].left() - 18:face[0].right() + 10]
                    # head = imutils.resize(head, width=500)
                    cv2.imwrite(photo_name, head)
                    print("new photo saved")
            TOTAL_0 = TOTAL
            out_model = f_liveness_detection.detect_liveness(im, COUNTER, TOTAL_0)
            TOTAL = out_model['total_blinks']
            print("total: ", TOTAL)

            COUNTER = out_model['count_blinks_consecutives']
            print("counter: ", COUNTER)
            dif_blink = TOTAL - TOTAL_0
            if dif_blink > 0:
                blinks_up = 1

            challenge_res = questions.challenge_result(question, out_model, blinks_up)

            # im = show_image(cam, question)
            # cv2.imshow('liveness_detection', im)
            if challenge_res == "pass":
                result = True
                print("result: ", result)
                # break
            elif challenge_res == "fail":
                counter_try += 1
                result = False
            print("result: ", result, "get_photo: ", get_photo)
            if result and get_photo:
                break
            if i_try == limit_try - 1:
                print("You have exceeded the number of attempts")
                break
        else:
            break
    # photo_name = "D:\\PycharmProjects\\GFPGAN\\results\\restored_imgs\\11.jpg"
    if result and get_photo:
        # upscale image with GFPGAN
        new_photo_dir = "D:\\PycharmProjects\\face_liveness_detection-Anti-spoofing\\upscaled"
        print(photo_name.split("\\")[-1])
        new_photo_name = new_photo_dir + "\\" + photo_name.split("\\")[-1]
        im = cv2.imread(photo_name)
        im = imutils.resize(im, width=1000)
        cv2.imwrite(new_photo_name, im)
        # async with aiohttp.ClientSession() as session:
        #     async with session.post('http://localhost:5000/upscale', json={"name": photo_name, "new_name":new_photo_dir,
        #                                                                    "upscale": 2}) as resp:
        #         data = await resp.json()
        #         if data['status'] == 'ok':
        #             im = cv2.imread(new_photo_name)
        #             im = imutils.resize(im, width=1000)
        #             cv2.imwrite(new_photo_name, im)
        #         else:
        #             result = False
        #             get_photo = False
        #             new_photo_name = ""
        # send image to server
    else:
        new_photo_name = ""
    # resize image

    return result, get_photo, new_photo_name


app = FastAPI()

class Video(BaseModel):
    name: str
    question: int


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/liveness")
async def check_liveness_view(video: Video):
    passed, get_photo, photo_name = await check_liveness(video.name, video.question)
    return {"check": passed, "get_photo": get_photo, "photo_name": photo_name}
