import numpy as np
import time
from face_detector import HaarDetector
from utils import utils
from face_tracker import MultiFaceTracker
from controller import Controller
from camera import Camera
from predictor import Predictor
from threading import Thread, active_count, Lock
import multiprocessing
from task_consumer import Consumer,Task
import cv2


class Pipeline:
    def __init__(self, config):
        self.controller = Controller(
            reset_limit_frame=config.FRAME_FOR_REDETECT, interval=config.INTERVAL
        )
        self.camera = Camera(config.CAMERA_FPS, self.controller)
        self.detector = HaarDetector()
        self.trackers = MultiFaceTracker(self.camera.get_frame(), [])
        self.predictor = Predictor(config.model, config.label)
        self.frame = []

        self.order_queue = multiprocessing.Queue()
        order_queue = [0]*config.FRAME_FOR_REDETECT + [1]
        for order in order_queue: 
            self.order_queue.put(order)

    def initialize_tracker(self):
        """
        Keep detecting until get some faces
        """
        while True:
            self.frame = self.camera.get_frame()
            faces = self.detector.detect(self.frame)
            if len(faces) > 0:
                for face in faces:
                    self.trackers.register(face, self.frame)
                print("FACE DETECTED")
                break

    def check_redetect(self):
        """
        If redetect_flag is triggered, count_frame in controller will be set to 0 and redetect face.
        """
        while True:

            if self.controller.trigger_redetect():
                self.controller.reset()
                # In-progress
                self.detector.detect(self.frame)
    
    def redetect(self,frame):
        faces = self.detector.detect(frame)
        print("Redetect got {} faces".format(len(faces)))

    def extract_image(self):
        """
        Classify stage: crop faces -> send to predictor -> get label and consumed time in inference stage.
        """
        while True:
            if self.controller.trigger_extract():
                cv2.imwrite("output/image.png", self.trackers.visualize(self.frame))
                faces = self.trackers.extract_face(self.frame)
                for idx in faces.keys():
                    print(self.predictor.classify_image(faces[idx]))

    def run(self):
        camera_thread = Thread(target=self.camera.start_camera).start()

        # Detect face for the first time
        initialize_thread = Thread(target=self.initialize_tracker)
        initialize_thread.start()
        initialize_thread.join()


        frame_queue = multiprocessing.Queue()
        task_queue = multiprocessing.JoinableQueue()

        consumer = Consumer(task_queue,frame_queue,self.trackers,self.detector)
        consumer.start()
        while True:
            frame_queue.put(self.camera.get_frame())

            # Remain order circle
            order = self.order_queue.get()
            self.order_queue.put(order)
            frame = frame_queue.get()
            task = Task(frame,self.trackers,self.detector,order)
            task_queue.put(task)


        # Extract faces and send to inference stage after a specific period, for ex (5 frame -> extract faces -> infer)
        extract_thread = Thread(target=self.extract_image).start()
