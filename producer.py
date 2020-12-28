import cv2
import multiprocessing
from threading import Thread

class Producer(multiprocessing.Process):
    """
    Handle camera input and store frames as a FIFO queue
    """

    def __init__(self, frame_queue, config, track_lock):
        multiprocessing.Process.__init__(self)
        self.frame_queue = frame_queue
        self.fps = 10


        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def start(self):
        """
        Start a new thread beyond main thread to solve blocking operation
        """
        t = Thread(target=self.update,args=())
        t.daemon= True
        t.start()
        return self 


    def update(self):
        while True:
            if not self.frame_queue.full():
                print("Buffer contains {} frames".format(self.frame_queue.qsize()))    
                ret,frame = self.cap.read()
                
                # ret flag to check if camere works properly or not
                if not ret:
                    return False
                self.frame_queue.put(frame)
                


    def release_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()