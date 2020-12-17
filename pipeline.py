from predictor import Predictor
import multiprocessing
from consumer import Consumer
from producer import Producer
import cv2


class Pipeline:
    def __init__(self, config):

        self.track_lock = multiprocessing.Lock()
        self.infer_lock = multiprocessing.Lock()

        self.frame_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()

        self.producer = Producer(self.frame_queue, config, self.track_lock)
        self.consumer = Consumer(
            self.frame_queue, self.result_queue, self.track_lock, self.infer_lock
        )
        self.predictor = Predictor(
            config.model, config.label, self.result_queue, self.infer_lock
        )

    def run(self):
        self.producer.start()
        self.consumer.start()
        self.predictor.start()
