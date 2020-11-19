import cv2
import numpy as np
import time
from collections import deque
import threading

class Camera:
    """
    Handle camera input and store frames as a FIFO queue
    """
    def __init__(self,fps,controller):
        self.frame = deque()
        self.fps = fps
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS,10)
        self.controller = controller

        # Initilize 50 first frame to avoid data races (temporary, need to fix)
        for _ in range(50):
            self.append_frame(self.cap.read()[1])
    
    def get_frame(self):
        return self.frame.popleft()
    
    def append_frame(self,frame):
        self.controller.increase_frame()
        self.frame.append(frame)
    
    def start_camera(self):
        """
        Limit the amount of frames in buffer always less than fps
        """
        while(True):
            print("Buffer contains {} frames".format(len(self.frame)))
            _,frame = self.cap.read()
            if len(self.frame) < self.fps:
                self.append_frame(frame)
                

    def release_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def visualize(frame):
        cv2.imwrite("output/1.png",frame)


