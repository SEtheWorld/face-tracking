import cv2
import multiprocessing
from controller import Controller


class Producer(multiprocessing.Process):
    """
    Handle camera input and store frames as a FIFO queue
    """

    def __init__(self, frame_queue, config, track_lock):
        multiprocessing.Process.__init__(self)
        self.frame = frame_queue
        self.fps = 10
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.controller = Controller(
            reset_limit_frame=config.FRAME_FOR_REDETECT, interval=config.INTERVAL
        )

    def get_additional_frame(self):
        _, frame = self.cap.read()
        self.append_frame(frame)

    def append_frame(self, frame):
        self.controller.increase_frame()
        self.frame.put(frame)

    def run(self):
        """
        Limit the amount of frames in buffer always less than fps
        """
        while True:
            print("Buffer contains {} frames".format(self.frame.qsize()))
            _, frame = self.cap.read()
            if self.frame.qsize() < self.fps:
                self.append_frame(frame)

    def release_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def visualize(frame):
        cv2.imwrite("output/1.png", frame)
