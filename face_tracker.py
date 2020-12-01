import cv2
from collections import OrderedDict
from utils.utils import get_iou, convert_box_to_int


class SingleFaceTracker:
    def __init__(self, frame, face):
        self.face = tuple(face)
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, self.face)

    def update(self, frame):
        _, updated_face = self.tracker.update(frame)
        iou = get_iou(updated_face, self.face)
        if iou > 0.7:
            self.face = updated_face
            disappeared = False
        else:
            self.face = tuple([0.0, 0.0, 0.0, 0.0])
            disappeared = True
        return self.face, disappeared


class MultiFaceTracker:
    def __init__(self, frame, input_faces, max_disappeared=20):

        self.faceID = 0
        self.face_trackers = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        for face in input_faces:
            self.register(frame, face)

    def register(self, face, frame):

        self.face_trackers[self.faceID] = SingleFaceTracker(frame, face)
        self.disappeared[self.faceID] = 0
        self.faceID += 1

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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.putText(
                        frame,str(faceID),
                        (x, y),cv2.FONT_HERSHEY_SIMPLEX,1,
                        (255, 0, 0),2,cv2.LINE_AA
                        )
        return frame
    
    def extract_face(self,frame):
        faces = dict()
        for faceID in list(self.face_trackers.keys()):
            (x,y,w,h) = convert_box_to_int(self.face_trackers[faceID].face)
            faces[faceID] = frame[x:x+w,y:y+h]
        return faces
