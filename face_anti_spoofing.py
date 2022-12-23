import random
import cv2
import imutils
import f_liveness_detection
import questions
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import os


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


def check_liveness(video_name, question_id):
    # Load the detector
    EYE_AR_THRESH = 0.35
    detector = dlib.get_frontal_face_detector()

    # Load the predictor
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    predictor = dlib.shape_predictor("blink_detection/model_landmarks/shape_predictor_68_face_landmarks.dat")
    cv2.namedWindow('liveness_detection')
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

    im = show_image(cam, question)
    cv2.imshow('liveness_detection', im)
    # if cv2.waitKey(1) &0xFF == ord('q'):
    #     break
    get_photo = False
    photo_name = ""
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
                if abs((nose_bot_right_ear_bot - nose_bot_left_ear_bot) / nose_bot_right_ear_bot) < 0.5 and \
                        abs((nose_bot_right_ear_up - nose_bot_left_ear_up) / nose_bot_right_ear_up) < 0.5 and \
                        ear >= EYE_AR_THRESH:
                    get_photo = True
                    photo_name = f"{video_name}_{COUNTER}.jpg"
                    print("photo_name: ", photo_name)
                    face_resized = imutils.resize(im, width=1200)
                    cv2.imwrite(photo_name, face_resized)

            # im = imutils.resize(im, width=720)
            # im = cv2.flip(im, 1)
            # <----------------------- ingestar data
            TOTAL_0 = TOTAL
            out_model = f_liveness_detection.detect_liveness(im, COUNTER, TOTAL_0)
            TOTAL = out_model['total_blinks']
            print("total: ", TOTAL)

            COUNTER = out_model['count_blinks_consecutives']
            print("counter: ", COUNTER)
            dif_blink = TOTAL - TOTAL_0
            if dif_blink > 0:
                blinks_up = 1
            else:
                blinks_up = 0
            print("blinks_up: ", blinks_up)
            challenge_res = questions.challenge_result(question, out_model, blinks_up)

            # im = show_image(cam, question)
            # cv2.imshow('liveness_detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if challenge_res == "pass":
                result = True
                break
                # break
                # im = show_image(cam, question + " : ok")
                # cv2.imshow('liveness_detection', im)


                # counter_ok_consecutives += 1
                # if counter_ok_consecutives == limit_consecutives:
                #     counter_ok_questions += 1
                #     counter_try = 0
                #     counter_ok_consecutives = 0
                #     break
                # else:
                #     continue

            elif challenge_res == "fail":
                counter_try += 1
                result = False
                # break
                # show_image(cam,question+" : fail")
            elif i_try == limit_try - 1:
                break
        else:
            break
    # video_capture.release()
    cv2.destroyAllWindows()
    return result, get_photo, photo_name

    # if counter_ok_questions ==  limit_questions:
    #     while True:
    #         im = show_image(cam,"LIFENESS SUCCESSFUL",color = (0,255,0))
    #         cv2.imshow('liveness_detection',im)
    #         if cv2.waitKey(1) &0xFF == ord('q'):
    #             break
    # elif i_try == limit_try-1:
    #     while True:
    #         im = show_image(cam,"LIFENESS FAIL")
    #         cv2.imshow('liveness_detection',im)
    #         if cv2.waitKey(1) &0xFF == ord('q'):
    #             break
    #     break

    # else:
    #     continue
