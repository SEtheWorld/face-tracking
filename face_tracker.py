import cv2
from collections import OrderedDict
from utils.utils import get_iou, convert_box_to_int
import numpy as np
import time
import random


IOU_THRESHOLD = 0.8
MIN_FACE_AREA = 2500
TIME_LIMIT = 50


class SingleFaceTracker:
    def __init__(self, frame, face):
        self.face = tuple(list(face))
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, self.face)
        self.frame = frame

    def update(self, frame):
        _, updated_face = self.tracker.update(frame)
        self.frame = frame
        iou = get_iou(updated_face, self.face)
        if iou > IOU_THRESHOLD:
            self.face = updated_face
            disappeared = False
        else:
            self.face = (0.0, 0.0, 0.0, 0.0)
            disappeared = True
        return self.face, disappeared

    def update_detect(self, detected_faces):
        """Update tracker for redetect function
            Delete face which can match with any face in tracker buffer

        Args:
            detected_faces ([type]): detected faces extract by Haar Cascade

        Returns:
            [Boolean]: True if any detected face can match with faces in buffer
        """
        for idx, detected_face in enumerate(detected_faces):
            iou = get_iou(detected_face, self.face)
            if iou > 0.6:
                self.face = detected_face
                detected_faces = np.delete(detected_faces, idx, axis=0)
        return detected_faces

    def get_face(self):

        (x, y, w, h) = convert_box_to_int(self.face)
        return self.frame[x : x + w, y : y + h]


class MultiFaceTracker:
    def __init__(self, frame, input_faces, result_queue, max_disappeared=20):

        self.faceID = 0
        self.face_trackers = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.result_queue = result_queue
        for face in input_faces:
            self.register(face, frame)

    def register(self, face, frame):
        (x, y, w, h) = face
        if w * h > MIN_FACE_AREA:
            self.face_trackers[self.faceID] = SingleFaceTracker(frame, face)
            self.disappeared[self.faceID] = 0
            self.faceID = int(time.time() * 10) % 10000

    def deregister(self, faceID):
        del self.face_trackers[faceID]
        del self.disappeared[faceID]

    def update(self, frame):
        for faceID in list(self.face_trackers.keys()):
            _, disappeared_flag = self.face_trackers[faceID].update(frame)
            if disappeared_flag:
                self.disappeared[faceID] += 1
                if self.disappeared[faceID] > self.max_disappeared:
                    self.deregister(faceID)
            else:
                self.disappeared[faceID] = 0

    def visualize(self, frame):
        for faceID in list(self.face_trackers.keys()):
            (x, y, w, h) = convert_box_to_int(self.face_trackers[faceID].face)
            # cv2.imwrite("output/{}.png".format(faceID),frame[x:x+w,y:y+h])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.putText(
                frame,
                str(faceID),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite("output/image.png", frame)

    def extract_face(self, frame):
        faces = {}
        for faceID in list(self.face_trackers.keys()):
            (x, y, w, h) = convert_box_to_int(self.face_trackers[faceID].face)
            faces[faceID] = frame[x : x + w, y : y + h]
        return faces

    def update_detect(self, frame, faces):
        for faceID in list(self.face_trackers.keys()):
            faces = self.face_trackers[faceID].update_detect(faces)
        for face in faces:
            self.register(face, frame)

    def get_result(self):
        anchor_time = int(time.time() * 10) % 10000
        for faceID in list(self.face_trackers.keys()):
            if anchor_time - faceID > TIME_LIMIT:
                self.result_queue.put(self.face_trackers[faceID].get_face())
