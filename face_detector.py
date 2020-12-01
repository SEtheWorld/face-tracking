from abc import ABC, abstractmethod
import cv2

class FaceDetector(ABC):
    @abstractmethod
    def detect(self, frame):
        raise NotImplementedError

class HaarDetector(FaceDetector):
    def __init__(self, cascPath="./model/haarcascade_frontalface_default.xml"):
        self.faceCascade = cv2.CascadeClassifier(cascPath)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
