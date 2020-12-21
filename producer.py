import cv2
import multiprocessing


class Producer(multiprocessing.Process):
    """
    Handle camera input and store frames as a FIFO queue
    """

    def __init__(self, frame_queue, config, track_lock):
        multiprocessing.Process.__init__(self)
        self.frame = frame_queue
        self.fps = 5
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def run(self):
        """
        Limit the amount of frames in buffer always less than fps
        """
        while True:
            print("Buffer contains {} frames".format(self.frame.qsize()))
            _, frame = self.cap.read()
            if self.frame.qsize() < self.fps:
                frame = cv2.resize(frame,(224,224))
                self.frame.put(frame)

    def release_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()