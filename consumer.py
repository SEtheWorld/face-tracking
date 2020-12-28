import multiprocessing
from face_detector import HaarDetector
from face_tracker import MultiFaceTracker
import cv2
import time
import logging

class Consumer(multiprocessing.Process):
    initialized = False
    # Track 4 frames -> redetect 1 frame
    def __init__(self, frame_queue, result_queue, track_lock, infer_lock):
        multiprocessing.Process.__init__(self)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.detector = HaarDetector()
        self.trackers = MultiFaceTracker([], [], result_queue=result_queue)
        self.track_lock = track_lock
        self.infer_lock = infer_lock
        self.count = 0
        logging.basicConfig(filename='track.log', filemode='w',level=logging.DEBUG)

    def initialize_tracker(self):
        """
        Keep detecting until get some faces
        """
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                faces = self.detector.detect(frame)
                if len(faces) > 0:
                    for face in faces:
                        self.trackers.register(face, frame)
                        print(face)
                    print("FACE DETECTED")
                    self.trackers.visualize(frame)
                    return True

    def run(self):
        if not Consumer.initialized:
            Consumer.initialized = self.initialize_tracker()
        start_time =time.time()
        time_track = time.time()
        time_detect = time.time()
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                # self.trackers.visualize(frame)
                if self.count < 15:
                    self.trackers.update(frame)
                    self.count += 1
                    logging.info("TRACK: {}".format((time.time()-time_track)*1000))
                    time_track = time.time()
                    
                else:
                    faces = self.detector.detect(frame)
                    self.trackers.update_detect(frame, faces)
                    self.count = 0
                    logging.info("REDETECT: {}".format((time.time()-time_detect)*1000))
                    time_detect = time.time()
                    logging.info("TRACK+DETECT: {}".format((time.time()-start_time)*1000))
                    start_time=time.time()
                    if self.result_queue.empty():
                        # with self.infer_lock:
                        self.trackers.get_result()
                    

