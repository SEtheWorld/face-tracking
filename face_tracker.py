import cv2
from collections import OrderedDict
from utils.utils import get_iou, convert_box_to_int
import numpy as np
IOU_THRESHOLD= 0.8 

class SingleFaceTracker:
    def __init__(self, frame, face):
        self.face = tuple(face)
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, self.face)

    def update(self, frame):
        _, updated_face = self.tracker.update(frame)
        iou = get_iou(updated_face, self.face)
        if iou > IOU_THRESHOLD:
            self.face = updated_face
            disappeared = False
        else:
            self.face = tuple([0.0, 0.0, 0.0, 0.0])
            disappeared = True
        return self.face, disappeared

    def update_detect(self,detected_faces):
        """Update tracker for redetect function
            Delete face which can match with any face in tracker buffer

        Args:
            detected_faces ([type]): detected faces extract by Haar Cascade

        Returns:
            [Boolean]: True if any detected face can match with faces in buffer
        """
        for idx,detected_face in enumerate(detected_faces):
            iou = get_iou(detected_face,self.face)
            if iou > IOU_THRESHOLD:
                self.face = detected_face
                detected_faces = np.delete(detected_faces,idx)
                return True
        return False




class MultiFaceTracker:
    def __init__(self, frame, input_faces, max_disappeared=20):

        self.faceID = 0
        self.face_trackers = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        for face in input_faces:
            self.register(face, frame)

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
            cv2.imwrite("output/{}.png".format(faceID),frame[x:x+w,y:y+h])
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
    
    def update_detect(self,frame,faces):
        for faceID in list(self.face_trackers.keys()):
            self.face_trackers[faceID].update_detect(faces)
        
        for face in faces:
            self.register(face,frame)
