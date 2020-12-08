import multiprocessing
from threading import Thread, Lock

class Consumer(multiprocessing.Process):
    
    # Track 4 frames -> redetect 1 frame
    def __init__(self,task_queue,frame_queue,trackers,detector):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.trackers = trackers
        self.detector = detector
        self.frame_queue =frame_queue
        self.lock = Lock()

    def run(self):
        task = self.task_queue.get()
        self.task_queue.put(task)
        with self.lock:
            frame = self.frame_queue.get()
            task.process(function)
            self.task_queue.task_done()
    
class Task():
    
    def __init__(self,frame,trackers,detector):
        self.frame = frame
        self.trackers = trackers
        self.detector = detector
    
    def process(self):
        self.function.update(self.frame)
        print("Process is doing {}".format(self.function))
    
    def track(self):
        self.trackers.update(self.frame)

    


