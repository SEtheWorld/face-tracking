import multiprocessing
import threading

class Consumer(threading.Thread):
    
    # Track 4 frames -> redetect 1 frame
    def __init__(self,task_queue,frame_queue,trackers,detector):
        threading.Thread.__init__(self)
        self.task_queue = task_queue
        self.trackers = trackers
        self.detector = detector
        self.frame_queue = frame_queue
        self.lock = threading.Lock()

    def run(self):
        while True:
            with self.lock:
                task = self.task_queue.get()
                self.task_queue.put(task)
                frame = self.frame_queue.get()
                if task.order == 0:
                    print("TRACK")
                    self.trackers.update(frame)
                else:
                    print("REDETECT")
                    faces = self.detector.detect(frame)
                    self.trackers.update_detect(frame,faces)
            self.task_queue.task_done()
    
class Task():
    
    def __init__(self,frame,order):
        self.frame = frame
        self.order = order

    def process(self):
        self.function.update(self.frame)
        print("Process is doing {}".format(self.function))

    


