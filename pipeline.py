import numpy as np
import time
from face_detector import HaarDetector
from utils import utils
from face_tracker import MultiFaceTracker
from controller import Controller
from camera import Camera
from threading import Thread,active_count
import cv2

class Pipeline:
    def __init__(self,config):
        self.controller = Controller(
            reset_limit_frame = config.FRAME_FOR_REDETECT, num_frame_capture=config.NUM_FRAME_CAPTURE
        )
        self.camera = Camera(config.CAMERA_FPS,self.controller)

        self.detector = HaarDetector()
        self.trackers = MultiFaceTracker(self.camera.get_frame(),[])

    def initialize_tracker(self):
        while True:
            frame = self.camera.get_frame()
            faces = self.detector.detect(frame)
            if len(faces) > 0:
                for face in faces:
                    self.trackers.register(face,frame)
                break
    
    def track(self,frame):
        while True:
            self.trackers.update(frame)

    def check_redetect(self):
        while True:
            # If redetect_flag is triggered, count_frame in controller will be set to 0 and redetect face
            if self.controller.trigger_redetect():
                self.controller.reset()
                # In-progress
                self.detector.detect(self.camera.get_frame())
    
    def run(self):
        camera_thread = Thread(target = self.camera.start_camera).start()
        
        # Detect face for the first time
        initialize_thread = Thread(target= self.initialize_tracker)
        initialize_thread.start()
        initialize_thread.join()

        detect_thread = Thread(target = self.check_redetect).start()
        track_thread = Thread(target=self.track,args=(self.camera.get_frame(),)).start()
        # detect_thread.join()

